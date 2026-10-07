import io
import re
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Try to initialize optional OCR backends (PaddleOCR, EasyOCR). Fall back to Tesseract.
OCR_ENGINE = "tesseract"
_paddle_ocr = None
_easy_reader = None
try:
    from paddleocr import PaddleOCR
    _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
    OCR_ENGINE = "paddle"
except Exception:
    _paddle_ocr = None
    try:
        import easyocr

        _easy_reader = easyocr.Reader(["en"], gpu=False)
        OCR_ENGINE = "easy"
    except Exception:
        _easy_reader = None
        OCR_ENGINE = "tesseract"


GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems."
)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).replace("\u2019", "'").upper()


def _preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    # Load into PIL then convert to OpenCV image (BGR)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert to gray, denoise, and enhance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2)
    return gray


def _deskew_image(img: np.ndarray) -> np.ndarray:
    # Estimate skew via edges and minimum area rectangle
    try:
        coords = np.column_stack(np.where(img < 255))
        if coords.shape[0] < 10:
            return img
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img


def _remove_glare(img: np.ndarray) -> np.ndarray:
    # Detect bright specular highlights and inpaint
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) < 10:
            return img
        inpainted = cv2.inpaint(img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask, 3, cv2.INPAINT_TELEA)
        if len(img.shape) == 2:
            return cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        return inpainted
    except Exception:
        return img


def _detect_text_regions(img: np.ndarray):
    # Basic morphological text region detection: gradient + closing then contours
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # Sobel gradient to highlight text
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = gray.shape[:2]
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # filter by size and aspect ratio
            if cw < 30 or ch < 10:
                continue
            if cw / max(ch, 1) < 1.0:  # likely not a horizontal text line
                continue
            # clip to image
            x2 = min(x + cw, w)
            y2 = min(y + ch, h)
            boxes.append((x, y, x2, y2))
        # sort by y then x
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes
    except Exception:
        return []


def _run_tesseract_on_image(cv_img: np.ndarray) -> str:
    # Keep wrapper name for tests; prefer PaddleOCR/EasyOCR if available
    return _ocr_recognize(cv_img)


def _ocr_recognize(cv_img: np.ndarray) -> str:
    """Generic OCR wrapper. Uses PaddleOCR > EasyOCR > Tesseract based on availability."""
    try:
        # Ensure we have an RGB or grayscale PIL image depending on engine
        if OCR_ENGINE == "paddle" and _paddle_ocr is not None:
            # PaddleOCR accepts image path or np.ndarray (BGR) — ensure BGR
            if len(cv_img.shape) == 2:
                img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR) if cv_img.shape[2] == 3 else cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            result = _paddle_ocr.ocr(img_bgr, cls=True)
            texts = []
            for line in result:
                # line: list of [box, (text, score)] or nested lists
                if isinstance(line, list):
                    for item in line:
                        if len(item) >= 2 and isinstance(item[1], tuple):
                            texts.append(item[1][0])
                        elif len(item) >= 2 and isinstance(item[1], str):
                            texts.append(item[1])
                elif isinstance(line, tuple) and len(line) >= 2:
                    texts.append(line[1][0] if isinstance(line[1], tuple) else line[1])
            return "\n".join(texts)

        if OCR_ENGINE == "easy" and _easy_reader is not None:
            # easyocr expects RGB images
            if len(cv_img.shape) == 2:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if cv_img.shape[2] == 3 else cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            texts = _easy_reader.readtext(img_rgb, detail=0)
            return "\n".join(texts)

        # Fallback to Tesseract
        pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB))
        text = pytesseract.image_to_string(pil)
        return text
    except Exception:
        # If any engine fails, try a best-effort tesseract fallback
        try:
            pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB))
            return pytesseract.image_to_string(pil)
        except Exception:
            return ""


def extract_fields_from_image(image_bytes: bytes) -> dict:
    """Run a simple OCR pipeline and extract a few label fields using regex/heuristics.

    This is intentionally small and heuristic-driven for prototype purposes.
    """
    gray = _preprocess_image_bytes(image_bytes)

    # If preprocessing returned something unexpected (e.g., test monkeypatch returns bytes), fall back
    if not isinstance(gray, np.ndarray):
        raw_text = _run_tesseract_on_image(gray)
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        text_full = "\n".join(lines)
    else:
        # deskew, remove glare
        img_ds = _deskew_image(gray)
        img_noglare = _remove_glare(img_ds)

        # detect text regions
        boxes = _detect_text_regions(img_noglare)
        region_texts = []
        if not boxes:
            # fallback to whole image OCR
            raw_text = _run_tesseract_on_image(img_noglare)
            region_texts = [raw_text]
        else:
            for (x1, y1, x2, y2) in boxes:
                crop = img_noglare[y1:y2, x1:x2]
                # upscale and enhance
                h, w = crop.shape[:2]
                scale = max(1, 800 // max(h, w))
                crop_rs = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                if len(crop_rs.shape) == 2:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    crop_rs = clahe.apply(crop_rs)
                raw = _run_tesseract_on_image(crop_rs)
                region_texts.append(raw)
        # join region texts in order
        lines = []
        for rt in region_texts:
            for l in rt.splitlines():
                if l.strip():
                    lines.append(l.strip())
        text_full = "\n".join(lines)

    # Brand: heuristics - assume the largest line near top; here we pick the first non-small line
    brand = lines[0] if lines else ""

    # Alcohol content (ABV) regex
    abv_match = re.search(r"(\d{1,2}(?:\.\d)?\s?%\s*(?:ALC\.?\/?VOL\.?)?)|(?:\d{1,3}\s?PROOF)", text_full, re.IGNORECASE)
    abv = abv_match.group(0) if abv_match else ""

    # Net contents
    net_match = re.search(r"\b(\d+(?:\.\d+)?\s*(?:ML|MILLILITER|L|LITER|FL\s?OZ))\b", text_full, re.IGNORECASE)
    net_contents = net_match.group(0) if net_match else ""

    # Government warning - normalize and compare prefix
    normalized = _normalize_text(text_full)
    gw_present = _normalize_text(GOVERNMENT_WARNING) in normalized

    # Class/Type: look for keywords like 'WHISKEY', 'BOURBON', 'VODKA', 'WINE', 'BEER'
    type_match = re.search(r"\b(BOURBON|WHISKEY|WHISKY|VODKA|GIN|WINE|BEER|DISTILLED SPIRITS|RUM)\b", text_full, re.IGNORECASE)
    class_type = type_match.group(0) if type_match else ""

    return {
        "raw_text": text_full,
        "brand": brand,
        "class_type": class_type,
        "alcohol_content": abv,
        "net_contents": net_contents,
        "government_warning_present": gw_present,
    }


def debug_extract_fields_from_image(image_bytes: bytes) -> dict:
    """Run extraction but also return diagnostic information (boxes, per-region OCR, engine).

    Useful for troubleshooting why an image produced empty results.
    Returns a dict with keys: fields (same as extract_fields...), engine, boxes, region_texts, overlay_image (base64)
    """
    import base64

    gray = _preprocess_image_bytes(image_bytes)
    diagnostics = {"engine": OCR_ENGINE, "boxes": [], "region_texts": []}

    if not isinstance(gray, np.ndarray):
        raw_text = _run_tesseract_on_image(gray)
        region_texts = [raw_text]
        boxes = []
        canvas = None
    else:
        img_color = None
        # keep a color version for overlay
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_color = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img_ds = _deskew_image(gray)
        img_noglare = _remove_glare(img_ds)

        boxes = _detect_text_regions(img_noglare)
        region_texts = []
        for (x1, y1, x2, y2) in boxes:
            crop = img_noglare[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            scale = max(1, 800 // max(h, w))
            crop_rs = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            if len(crop_rs.shape) == 2:
                try:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    crop_rs = clahe.apply(crop_rs)
                except Exception:
                    pass
            raw = _run_tesseract_on_image(crop_rs)
            region_texts.append(raw)

        # build overlay image showing boxes
        canvas = img_color.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    diagnostics["boxes"] = boxes
    diagnostics["region_texts"] = region_texts

    # Encode overlay image to base64 if available
    overlay_b64 = None
    if 'canvas' in locals() and canvas is not None:
        _, buf = cv2.imencode('.jpg', canvas)
        overlay_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    # Reuse main extractor to compute field values
    fields = extract_fields_from_image(image_bytes)

    return {
        "fields": fields,
        "engine": diagnostics["engine"],
        "boxes": diagnostics["boxes"],
        "region_texts": diagnostics["region_texts"],
        "overlay_image_base64": overlay_b64,
    }
