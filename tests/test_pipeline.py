import json

import pytest

from novel_translator.glossary import DEFAULT_GLOSSARY
from novel_translator.ner import EntityCandidate
import novel_translator.pipeline as pipeline_module
from novel_translator.pipeline import (
    TermBatchError,
    TranslationPipeline,
    context_candidate_batches,
    glossary_violations,
    residual_glossary_sources,
    split_context_blocks,
    split_text,
    translation_limits,
    translation_quality_issues,
)


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(
        self,
        model,
        system,
        prompt,
        format_schema=None,
        options=None,
        timeout=None,
    ):
        self.calls.append(
            (model, system, prompt, format_schema, options, timeout)
        )
        return next(self.responses)


class FailingClient(StubClient):
    def generate(
        self,
        model,
        system,
        prompt,
        format_schema=None,
        options=None,
        timeout=None,
    ):
        self.calls.append(
            (model, system, prompt, format_schema, options, timeout)
        )
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


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
    flow = TranslationPipeline(client, ner_engine="spacy")
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
    flow = TranslationPipeline(client, ner_engine="spacy")
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
        StubClient([json.dumps(proposal, ensure_ascii=False)]),
        ner_engine="spacy",
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
    repaired = {"target": "桃樂絲"}
    flow = TranslationPipeline(
        StubClient(
            [
                json.dumps(classified, ensure_ascii=False),
                json.dumps(repaired, ensure_ascii=False),
            ]
        ),
        ner_engine="spacy",
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


def test_enforce_glossary_replaces_english_without_model_call():
    glossary = """| 原文名稱 | 中文譯名 | 類型 | 首次出現章節 | 備註 |
| --- | --- | --- | --- | --- |
| Dorothy | 桃樂絲 | 人物 | Chapter 1 | |
"""
    client = StubClient([])
    flow = TranslationPipeline(client)
    result, violations = flow.enforce_glossary(
        "Dorothy arrived.", "Dorothy 抵達了。", glossary
    )
    assert result == "桃樂絲抵達了。"
    assert violations == []
    assert client.calls == []


def test_translation_retries_when_first_result_omits_body():
    incomplete = json.dumps(
        {"translation": "第一章\n旋風"}, ensure_ascii=False
    )
    complete = json.dumps(
        {
            "translation": (
                "第一章\n旋風\n\n桃樂絲住在廣闊的堪薩斯草原中央，"
                "與亨利叔叔及艾姆嬸嬸同住。"
            )
        },
        ensure_ascii=False,
    )
    source = (
        "Chapter I\nThe Cyclone\n\nDorothy lived in the midst of the great "
        "Kansas prairies, with Uncle Henry and Aunt Em."
    )
    client = StubClient([incomplete, complete])
    result = TranslationPipeline(client).translate(source, DEFAULT_GLOSSARY)
    assert "桃樂絲住在" in result
    assert len(client.calls) == 2
    assert "上一版譯文不合格" in client.calls[1][2]
    assert "第一章\n旋風" not in client.calls[1][2]
    assert client.calls[0][4]["num_predict"] >= 256
    assert 120 <= client.calls[0][5] <= 420


def test_translation_rejects_glossary_instead_of_translation():
    source = "Dorothy lived in Kansas with Uncle Henry and Aunt Em."
    output = "# 小說術語表\n| 原文名稱 | 中文譯名 |"
    issues = translation_quality_issues(source, output)
    assert "輸出了術語表而非小說譯文" in issues


def test_translation_limits_scale_with_source_length():
    short_options, short_timeout = translation_limits("Dorothy arrived.")
    long_options, long_timeout = translation_limits("word " * 1000)
    assert short_options["num_predict"] == 256
    assert short_timeout == 146
    assert long_options["num_predict"] == 2048
    assert long_timeout == 420


def test_reports_english_glossary_source_left_in_review():
    glossary = """| 原文名稱 | 中文譯名 | 類型 | 首次出現章節 | 備註 |
| --- | --- | --- | --- | --- |
| Dorothy | 桃樂絲 | 人物 | Chapter 1 | |
"""
    residual = residual_glossary_sources(
        "Dorothy arrived.",
        "桃樂絲看見 Dorothy 抵達了。",
        glossary,
    )
    assert residual == ["Dorothy"]


def test_scan_batches_candidates_and_uses_local_context(monkeypatch):
    candidates = [
        EntityCandidate(f"Name{index}", "人物", "test")
        for index in range(10)
    ]
    monkeypatch.setattr(
        pipeline_module, "extract_candidates", lambda source, engine: candidates
    )
    source = (
        "First shared paragraph introduces "
        + ", ".join(f"Name{index}" for index in range(8))
        + ".\n\nSecond shared paragraph introduces Name8 and Name9."
    )
    responses = []
    for size in (8, 2):
        responses.append(
            json.dumps(
                {
                    f"entity_{index}": {
                        "target": f"名稱{index}",
                        "type": "人物",
                        "remarks": "",
                    }
                    for index in range(size)
                },
                ensure_ascii=False,
            )
        )
    client = StubClient(responses)
    flow = TranslationPipeline(client, term_batch_size=8)
    terms = flow.scan(source, DEFAULT_GLOSSARY, "Chapter 1")
    assert len(terms) == 10
    assert flow.last_scan_batch_count == 2
    assert len(client.calls) == 2
    assert "Name0" in client.calls[0][2]
    assert "Name8" not in client.calls[0][2]
    assert "Name8" in client.calls[1][2]
    assert "Name0" not in client.calls[1][2]


def test_scan_batch_failure_preserves_completed_terms(monkeypatch):
    candidates = [
        EntityCandidate(f"Name{index}", "人物", "test")
        for index in range(3)
    ]
    monkeypatch.setattr(
        pipeline_module, "extract_candidates", lambda source, engine: candidates
    )
    first_batch = json.dumps(
        {
            f"entity_{index}": {
                "target": f"名稱{index}",
                "type": "人物",
                "remarks": "",
            }
            for index in range(2)
        },
        ensure_ascii=False,
    )
    flow = TranslationPipeline(
        FailingClient([first_batch, TimeoutError("逾時")]),
        term_batch_size=2,
    )
    with pytest.raises(TermBatchError) as caught:
        flow.scan(
            "Name0, Name1, and Name2 appeared together.",
            DEFAULT_GLOSSARY,
            "Chapter 1",
        )
    assert caught.value.batch_number == 2
    assert [term.source for term in caught.value.partial_terms] == [
        "Name0",
        "Name1",
    ]


def test_context_blocks_use_paragraph_sentence_and_maximum_length():
    source = (
        "Short opening paragraph.\n\n"
        + ("First long sentence words. " * 30)
        + ("X" * 120)
    )
    blocks = split_context_blocks(source, max_chars=100)
    assert blocks[0] == "Short opening paragraph."
    assert all(len(block) <= 100 for block in blocks)
    assert any("X" * 80 in block for block in blocks)


def test_candidates_are_grouped_by_shared_context_block():
    candidates = [
        EntityCandidate("Dorothy", "人物", "test"),
        EntityCandidate("Kansas", "地名", "test"),
        EntityCandidate("Toto", "人物", "test"),
    ]
    source = (
        "Dorothy lived in Kansas.\n\n"
        "Much later, Toto entered the room."
    )
    batches = context_candidate_batches(source, candidates, batch_size=8)
    assert len(batches) == 2
    assert [item.text for item in batches[0][1]] == ["Dorothy", "Kansas"]
    assert [item.text for item in batches[1][1]] == ["Toto"]


def test_empty_target_and_type_are_repaired_without_failing_batch(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "extract_candidates",
        lambda source, engine: [EntityCandidate("Dorothy", "人物", "test")],
    )
    incomplete = json.dumps(
        {
            "entity_0": {
                "target": "",
                "type": "",
                "remarks": "",
            }
        }
    )
    repaired = json.dumps({"target": "桃樂絲"}, ensure_ascii=False)
    flow = TranslationPipeline(StubClient([incomplete, repaired]))
    terms = flow.scan("Dorothy arrived.", DEFAULT_GLOSSARY, "Chapter 1")
    assert terms[0].target == "桃樂絲"
    assert terms[0].type == "人物"


def test_failed_single_target_repair_keeps_visible_placeholder(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "extract_candidates",
        lambda source, engine: [EntityCandidate("Dorothy", "人物", "test")],
    )
    incomplete = json.dumps(
        {
            "entity_0": {
                "target": "",
                "type": "人物",
                "remarks": "",
            }
        }
    )
    flow = TranslationPipeline(
        FailingClient([incomplete, TimeoutError("逾時")])
    )
    terms = flow.scan("Dorothy arrived.", DEFAULT_GLOSSARY, "Chapter 1")
    assert terms[0].target == "（待人工確認）"
    assert "請人工修改" in terms[0].remarks
