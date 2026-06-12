import json

from novel_translator.glossary import DEFAULT_GLOSSARY
from novel_translator.pipeline import (
    TranslationPipeline,
    glossary_violations,
    split_text,
)


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, model, system, prompt, format_schema=None):
        self.calls.append((model, system, prompt, format_schema))
        return next(self.responses)


def test_scan_translate_and_review_flow():
    proposal = json.dumps(
        {
            "entity_0": {
                "target": "艾莉絲",
                "type": "人物",
                "remarks": "",
            }
        },
        ensure_ascii=False,
    )
    translated = json.dumps({"translation": "艾莉絲抵達了。"}, ensure_ascii=False)
    client = StubClient([proposal, translated, translated])
    flow = TranslationPipeline(client)
    assert flow.scan("Iris arrived.", DEFAULT_GLOSSARY, "Chapter 1")
    draft = flow.translate("Iris arrived.", DEFAULT_GLOSSARY)
    assert flow.review("Iris arrived.", draft, DEFAULT_GLOSSARY) == draft
    assert len(client.calls) == 3


def test_scan_prompt_contains_hybrid_ner_candidates():
    proposal = {
        f"entity_{index}": {"target": "測試", "type": "人物", "remarks": ""}
        for index in range(4)
    }
    client = StubClient([json.dumps(proposal, ensure_ascii=False)])
    flow = TranslationPipeline(client)
    flow.scan(
        "Dorothy lived in Kansas with Uncle Henry and Aunt Em.",
        DEFAULT_GLOSSARY,
        "Chapter 1",
    )
    prompt = client.calls[0][2]
    assert "Dorothy" in prompt
    assert "Kansas" in prompt
    assert "Uncle Henry" in prompt
    assert "Aunt Em" in prompt


def test_scan_preserves_every_spacy_candidate():
    proposal = {
        "entity_0": {"target": "桃樂絲", "type": "人物", "remarks": ""},
        "entity_1": {"target": "堪薩斯州", "type": "地名", "remarks": ""},
        "entity_2": {"target": "亨利叔叔", "type": "稱謂", "remarks": ""},
        "entity_3": {"target": "艾姆阿姨", "type": "稱謂", "remarks": ""},
    }
    flow = TranslationPipeline(
        StubClient([json.dumps(proposal, ensure_ascii=False)])
    )
    terms = flow.scan(
        "Dorothy lived in Kansas with Uncle Henry and Aunt Em.",
        DEFAULT_GLOSSARY,
        "Chapter 1",
    )
    assert [term.source for term in terms] == [
        "Dorothy",
        "Kansas",
        "Uncle Henry",
        "Aunt Em",
    ]


def test_scan_repairs_targets_that_are_still_english():
    classified = {
        "entity_0": {"target": "Dorothy", "type": "人物", "remarks": ""},
    }
    repaired = {"target_0": "桃樂絲"}
    flow = TranslationPipeline(
        StubClient(
            [
                json.dumps(classified, ensure_ascii=False),
                json.dumps(repaired, ensure_ascii=False),
            ]
        )
    )
    terms = flow.scan("Dorothy arrived.", DEFAULT_GLOSSARY, "Chapter 1")
    assert terms[0].target == "桃樂絲"


def test_long_text_is_split_without_loss():
    source = "First paragraph.\n\nSecond paragraph."
    chunks = split_text(source, 18)
    assert chunks == ["First paragraph.", "Second paragraph."]


def test_reports_missing_required_glossary_target():
    glossary = """| 原文名稱 | 中文譯名 | 類型 | 首次出現章節 | 備註 |
| --- | --- | --- | --- | --- |
| Dorothy | 桃樂絲 | 人物 | Chapter 1 | |
"""
    assert glossary_violations("Dorothy arrived.", "多蘿西抵達了。", glossary) == [
        "Dorothy → 桃樂絲"
    ]


def test_enforce_glossary_corrects_before_reporting():
    glossary = """| 原文名稱 | 中文譯名 | 類型 | 首次出現章節 | 備註 |
| --- | --- | --- | --- | --- |
| Dorothy | 桃樂絲 | 人物 | Chapter 1 | |
"""
    corrected = json.dumps({"translation": "桃樂絲抵達了。"}, ensure_ascii=False)
    client = StubClient([corrected])
    flow = TranslationPipeline(client)
    result, violations = flow.enforce_glossary(
        "Dorothy arrived.", "多蘿西抵達了。", glossary
    )
    assert result == "桃樂絲抵達了。"
    assert violations == []
