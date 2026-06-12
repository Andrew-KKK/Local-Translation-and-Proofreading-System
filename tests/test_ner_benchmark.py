from novel_translator.ner_benchmark import (
    Entity,
    ExtractionResult,
    compare_extractions,
    predict_gliner_chunks,
    score,
    split_text_chunks,
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


def test_split_text_chunks_limits_size_and_adds_overlap():
    text = " ".join(f"word{index}" for index in range(10))
    chunks = split_text_chunks(text, chunk_size=4, overlap=1)
    assert chunks == [
        "word0 word1 word2 word3",
        "word3 word4 word5 word6",
        "word6 word7 word8 word9",
    ]


def test_gliner_processes_later_chunks_and_deduplicates_overlap():
    class FakeModel:
        def predict_entities(self, text, labels, threshold):
            predictions = []
            if "Dorothy" in text:
                predictions.append({"text": "Dorothy", "label": "person"})
            if "Toto" in text:
                predictions.append({"text": "Toto", "label": "person"})
            return predictions

    text = "Dorothy one two three Toto five six"
    entities = predict_gliner_chunks(
        FakeModel(), text, threshold=0.5, chunk_size=4, chunk_overlap=1
    )
    assert entities == [
        Entity("Dorothy", "人物"),
        Entity("Toto", "人物"),
    ]
