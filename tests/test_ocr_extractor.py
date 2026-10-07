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


def test_warning_matches_lowercase_and_ocr_noise():
    text = (
        "Government Warning: (1) according to the surgeon general, women should not drink "
        "alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your abi1ity to drive a car or operate "
        "machinery, and may cause heaIth problems."
    )
    assert ocr_extractor._match_government_warning(text) >= ocr_extractor.WARNING_MATCH_THRESHOLD


def test_warning_matches_with_surrounding_text_and_odd_spacing():
    text = "OLD TOM\n750 mL\nGOVERNMENT WARNING (1) ACCORDING TO THE SURGEON GENERAL WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY AND MAY CAUSE HEALTH PROBLEMS\nBottled by X"
    assert ocr_extractor._match_government_warning(text) >= ocr_extractor.WARNING_MATCH_THRESHOLD


def test_warning_absent_or_unrelated_text_does_not_match():
    assert ocr_extractor._match_government_warning("OLD TOM DISTILLERY 45% Alc./Vol. 750 mL") < ocr_extractor.WARNING_MATCH_THRESHOLD
    assert ocr_extractor._match_government_warning("") == 0.0
