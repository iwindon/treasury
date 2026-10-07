"""EasyOCR-only OCR extractor for label images.

This module provides extract_fields_from_image and debug_extract_fields_from_image
used by the FastAPI app. It is intentionally simple and uses EasyOCR when
available; otherwise OCR calls return empty strings.
"""

import io
import re
import base64
import logging
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger("uvicorn.error")

# Lazy EasyOCR import to avoid heavy model download on module import during tests
_easy_reader = None
_easy_reader_initialized = False


def _get_easy_reader():
    """Lazily import and initialize easyocr.Reader. Returns None on failure."""
    global _easy_reader, _easy_reader_initialized
    if _easy_reader_initialized:
        return _easy_reader
    _easy_reader_initialized = True
    try:
        import easyocr

        _easy_reader = easyocr.Reader(["en"], gpu=False)
    except Exception:
        logger.exception("EasyOCR reader failed to initialize")
        _easy_reader = None
    return _easy_reader


GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems."
)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).replace("\u2019", "'").upper()


def _preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image bytes and perform light preprocessing: RGB->BGR, gray, denoise, threshold."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    try:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 2)
    except Exception:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray


def _deskew_image(img: np.ndarray) -> np.ndarray:
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


def _detect_text_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = gray.shape[:2]
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 30 or ch < 10:
                continue
            if cw / max(ch, 1) < 1.0:
                continue
            x2 = min(x + cw, w)
            y2 = min(y + ch, h)
            boxes.append((x, y, x2, y2))
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes
    except Exception:
        return []


def _ocr_recognize(cv_img: np.ndarray) -> str:
    """Recognize text using EasyOCR (lazy init). Returns joined lines or empty string."""
    reader = _get_easy_reader()
    if reader is None:
        return ""
    try:
        if len(cv_img.shape) == 2:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if cv_img.shape[2] == 3 else cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        texts = reader.readtext(img_rgb, detail=0, paragraph=False)
        return "\n".join(texts)
    except Exception:
        logger.exception("EasyOCR readtext failed")
        return ""


def _ocr_original_image(image_bytes: bytes) -> str:
    """OCR the untouched color image; EasyOCR's own detector works best without binarization."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest < 1200:
        scale = 1200 / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    elif longest > 3000:
        scale = 3000 / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return _ocr_recognize(img)


def _run_tesseract_on_image(cv_img: np.ndarray) -> str:
    """Compatibility wrapper name used by tests. Uses EasyOCR under the hood."""
    return _ocr_recognize(cv_img)


def extract_fields_from_image(image_bytes: bytes) -> dict:
    """Run OCR pipeline and extract a few label fields using regex heuristics."""
    gray = _preprocess_image_bytes(image_bytes)

    if not isinstance(gray, np.ndarray):
        raw_text = _run_tesseract_on_image(gray)
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        text_full = "\n".join(lines)
    else:
        img_ds = _deskew_image(gray)
        img_noglare = _remove_glare(img_ds)
        full_text = _ocr_original_image(image_bytes)
        boxes = [] if full_text.strip() else _detect_text_regions(img_noglare)
        region_texts: List[str] = []
        if full_text.strip():
            region_texts = [full_text]
        elif not boxes:
            raw_text = _run_tesseract_on_image(img_noglare)
            region_texts = [raw_text]
        else:
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
        lines = []
        for rt in region_texts:
            for l in rt.splitlines():
                if l.strip():
                    lines.append(l.strip())
        text_full = "\n".join(lines)

    brand = lines[0] if lines else ""

    abv_match = re.search(r"(\d{1,2}(?:\.\d)?\s?%\s*(?:ALC\.?\/?VOL\.? )?)|(?:\d{1,3}\s?PROOF)", text_full, re.IGNORECASE)
    abv = abv_match.group(0) if abv_match else ""

    net_match = re.search(r"\b(\d+(?:\.\d+)?\s*(?:ML|MILLILITER|L|LITER|FL\s?OZ))\b", text_full, re.IGNORECASE)
    net_contents = net_match.group(0) if net_match else ""

    normalized = _normalize_text(text_full)
    gw_present = _normalize_text(GOVERNMENT_WARNING) in normalized

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
    """Return fields plus diagnostics: engine, boxes, region_texts, overlay image base64."""
    gray = _preprocess_image_bytes(image_bytes)
    diagnostics = {"engine": "easyocr", "boxes": [], "region_texts": []}

    if not isinstance(gray, np.ndarray):
        raw_text = _run_tesseract_on_image(gray)
        region_texts = [raw_text]
        boxes = []
        canvas = None
    else:
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

        canvas = img_color.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    diagnostics["boxes"] = boxes
    diagnostics["region_texts"] = region_texts

    overlay_b64 = None
    if 'canvas' in locals() and canvas is not None:
        _, buf = cv2.imencode('.jpg', canvas)
        overlay_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    fields = extract_fields_from_image(image_bytes)

    return {
        "fields": fields,
        "engine": diagnostics["engine"],
        "boxes": diagnostics["boxes"],
        "region_texts": diagnostics["region_texts"],
        "overlay_image_base64": overlay_b64,
    }

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
