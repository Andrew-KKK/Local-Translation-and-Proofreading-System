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
        [{"source": "Iris", "target": "艾莉絲", "type": "人物"}],
        ensure_ascii=False,
    )
    translated = json.dumps({"translation": "艾莉絲抵達了。"}, ensure_ascii=False)
    client = StubClient([proposal, translated, translated])
    flow = TranslationPipeline(client)
    assert flow.scan("Iris arrived.", DEFAULT_GLOSSARY, "Chapter 1")
    draft = flow.translate("Iris arrived.", DEFAULT_GLOSSARY)
    assert flow.review("Iris arrived.", draft, DEFAULT_GLOSSARY) == draft
    assert len(client.calls) == 3


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
