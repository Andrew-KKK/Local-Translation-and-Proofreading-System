from novel_translator.ner_benchmark import (
    Entity,
    ExtractionResult,
    compare_extractions,
    score,
)


def test_score_separates_detection_from_type_error():
    predicted = [
        Entity("Dorothy", "人物"),
        Entity("Uncle Henry", "人物"),
        Entity("Cyclone", "其他專名"),
    ]
    gold = [
        Entity("Dorothy", "人物"),
        Entity("Uncle Henry", "稱謂"),
        Entity("Kansas", "地名"),
    ]
    result = score("test", 0.1, predicted, gold)
    assert result.true_positive == ["dorothy", "uncle henry"]
    assert result.false_positive == ["cyclone"]
    assert result.false_negative == ["kansas"]
    assert result.type_errors == [
        "uncle henry: 預測 人物，正確 稱謂"
    ]
    assert round(result.precision, 3) == 0.667
    assert round(result.recall, 3) == 0.667


def test_compares_extracted_terms_without_gold_file():
    results = [
        ExtractionResult(
            "a", 0.1, [Entity("Dorothy", "人物"), Entity("Kansas", "地名")]
        ),
        ExtractionResult(
            "b", 0.2, [Entity("Dorothy", "人物"), Entity("Toto", "人物")]
        ),
    ]
    comparison = compare_extractions(results)[0]
    assert comparison["common"] == ["dorothy"]
    assert comparison["left_only"] == ["kansas"]
    assert comparison["right_only"] == ["toto"]
