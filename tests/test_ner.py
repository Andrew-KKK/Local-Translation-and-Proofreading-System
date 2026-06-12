from novel_translator.ner import extract_candidates


def test_hybrid_ner_extracts_oz_entities():
    text = """Chapter I
The Cyclone

Dorothy lived in Kansas with Uncle Henry and Aunt Em."""
    values = {item.text for item in extract_candidates(text)}
    assert {"Dorothy", "Kansas", "Uncle Henry", "Aunt Em"} <= values
    assert "Chapter" not in values


def test_title_rules_suggest_title_type():
    candidates = {item.text: item for item in extract_candidates("Aunt Em smiled.")}
    assert candidates["Aunt Em"].suggested_type == "稱謂"


def test_title_candidate_suppresses_split_person_parts():
    text = "Dorothy lived with Uncle\nHenry and Aunt Em."
    values = {item.text for item in extract_candidates(text)}
    assert "Dorothy" in values
    assert "Uncle Henry" in values
    assert "Uncle" not in values
    assert "Henry" not in values
