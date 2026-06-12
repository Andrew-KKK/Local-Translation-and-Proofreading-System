import novel_translator.ner as ner
from novel_translator.ner import extract_candidates


def test_hybrid_ner_extracts_oz_entities():
    text = """Chapter I
The Cyclone

Dorothy lived in Kansas with Uncle Henry and Aunt Em."""
    values = {item.text for item in extract_candidates(text, "spacy")}
    assert {"Dorothy", "Kansas", "Uncle Henry", "Aunt Em"} <= values
    assert "Chapter" not in values


def test_title_rules_suggest_title_type():
    candidates = {
        item.text: item
        for item in extract_candidates("Aunt Em smiled.", "spacy")
    }
    assert candidates["Aunt Em"].suggested_type == "稱謂"


def test_title_candidate_suppresses_split_person_parts():
    text = "Dorothy lived with Uncle\nHenry and Aunt Em."
    values = {item.text for item in extract_candidates(text, "spacy")}
    assert "Dorothy" in values
    assert "Uncle Henry" in values
    assert "Uncle" not in values
    assert "Henry" not in values


def test_gliner_is_default_and_merges_overlapping_chunk_results(monkeypatch):
    class FakeModel:
        def predict_entities(self, text, labels, threshold):
            values = []
            if "Dorothy" in text:
                values.append({"text": "Dorothy", "label": "person"})
            if "Toto" in text:
                values.append({"text": "Toto", "label": "person"})
            return values

    monkeypatch.setattr(ner, "_load_gliner_model", lambda: FakeModel())
    monkeypatch.setattr(ner, "DEFAULT_GLINER_CHUNK_SIZE", 4)
    monkeypatch.setattr(ner, "DEFAULT_GLINER_CHUNK_OVERLAP", 1)
    values = {
        item.text
        for item in extract_candidates(
            "Dorothy one two three Toto five six"
        )
    }
    assert values == {"Dorothy", "Toto"}
