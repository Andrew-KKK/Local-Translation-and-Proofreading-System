import json

import pytest

from novel_translator.glossary import (
    Term,
    merge_terms,
    parse_glossary,
    parse_proposals,
    render_glossary,
)


def test_markdown_round_trip():
    terms = [Term("Iris", "艾莉絲", "人物", "Chapter 1", "信使")]
    assert parse_glossary(render_glossary(terms)) == terms


def test_approved_term_updates_existing_entry():
    old = [Term("Iris", "艾莉絲", "人物")]
    new = [Term("iris", "伊莉絲", "人物")]
    result = merge_terms(old, new)
    assert len(result) == 1
    assert result[0].target == "伊莉絲"


def test_model_json_is_validated():
    data = [{"source": "Iris", "target": "艾莉絲", "type": "人物"}]
    assert parse_proposals(json.dumps(data))[0].source == "Iris"
    with pytest.raises(ValueError):
        parse_proposals("not json")
