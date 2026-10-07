from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from typing import List
from ocr_extractor import extract_fields_from_image
import uvicorn
from pathlib import Path


app = FastAPI(title="Label OCR Prototype")


@app.post("/upload")
async def upload_label(image: UploadFile = File(...), debug: bool = False):
    """Receive an image file, run OCR + extraction, and return parsed fields.

    If debug=true is provided, return diagnostic information (detected boxes, per-region OCR, overlay image base64).
    """
    contents = await image.read()
    try:
        if debug:
            # import lazily to avoid unnecessary work in non-debug mode
            from ocr_extractor import debug_extract_fields_from_image

            results = debug_extract_fields_from_image(contents)
            return JSONResponse(content={"success": True, "data": results})
        else:
            results = extract_fields_from_image(contents)
            return JSONResponse(content={"success": True, "data": results})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/upload/batch")
async def upload_batch(images: List[UploadFile] = File(...)):
    """Accept multiple image files and return an array of extraction results."""
    results = []
    for image in images:
        contents = await image.read()
        try:
            data = extract_fields_from_image(contents)
            results.append({"filename": image.filename, "data": data})
        except Exception as e:
            results.append({"filename": image.filename, "error": str(e)})
    return JSONResponse(content={"success": True, "data": results})


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>Label OCR Prototype</h1><p>No UI available.</p>", status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
