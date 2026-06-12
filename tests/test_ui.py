from novel_translator.glossary import DEFAULT_GLOSSARY, parse_glossary
from novel_translator.ui import (
    glossary_with_rows,
    prepare_translation_glossary,
)


def test_candidate_rows_are_merged_when_translation_starts():
    rows = [
        ["Dorothy", "桃樂絲", "人物", "Chapter 1", ""],
        ["Kansas", "堪薩斯州", "地名", "Chapter 1", ""],
    ]
    updated = glossary_with_rows(rows, DEFAULT_GLOSSARY)
    terms = parse_glossary(updated)
    assert [(term.source, term.target) for term in terms] == [
        ("Dorothy", "桃樂絲"),
        ("Kansas", "堪薩斯州"),
    ]


def test_prepare_translation_glossary_updates_before_translation():
    rows = [["Dorothy", "桃樂絲", "人物", "Chapter 1", ""]]
    updated, status = prepare_translation_glossary(rows, DEFAULT_GLOSSARY)
    assert parse_glossary(updated)[0].target == "桃樂絲"
    assert "正在產生翻譯初稿" in status
