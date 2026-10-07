import ocr_extractor


def test_extract_fields_basic(monkeypatch):
    sample_text = (
        "OLD TOM DISTILLERY\n"
        "Kentucky Straight Bourbon Whiskey\n"
        "45% Alc./Vol. (90 Proof)\n"
        "750 mL\n"
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems.\n"
    )

    # Monkeypatch the internal OCR functions to return our sample text and avoid real image processing
    monkeypatch.setattr(ocr_extractor, "_preprocess_image_bytes", lambda b: b)
    monkeypatch.setattr(ocr_extractor, "_run_tesseract_on_image", lambda img: sample_text)
    # Ensure region detection is bypassed in test by returning a single full-image box
    monkeypatch.setattr(ocr_extractor, "_detect_text_regions", lambda img: [(0, 0, 100, 100)])

    result = ocr_extractor.extract_fields_from_image(b"dummy-bytes")

    assert "OLD TOM DISTILLERY" in result["raw_text"]
    assert result["brand"].upper().startswith("OLD TOM")
    assert "BOURBON" in result["class_type"].upper() or "WHISKEY" in result["class_type"].upper()
    assert "45" in result["alcohol_content"]
    assert "750" in result["net_contents"]
    assert result["government_warning_present"] is True
