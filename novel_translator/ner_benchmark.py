from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Callable


LABELS = {
    "person": "人物",
    "location": "地名",
    "organization": "組織",
    "fictional object": "物件",
    "ability": "能力",
    "title": "稱謂",
    "other proper noun": "其他專名",
}
SPACY_LABELS = {
    "PERSON": "人物",
    "GPE": "地名",
    "LOC": "地名",
    "ORG": "組織",
    "PRODUCT": "物件",
    "EVENT": "其他專名",
    "WORK_OF_ART": "其他專名",
}
CHAPTER_PATTERN = re.compile(
    r"^(?:chapter|book|part)\s+(?:\d+|[ivxlcdm]+)$", re.IGNORECASE
)
DEFAULT_GLINER_CHUNK_SIZE = 220
DEFAULT_GLINER_CHUNK_OVERLAP = 40


@dataclass(frozen=True)
class Entity:
    text: str
    type: str


@dataclass
class MethodResult:
    method: str
    seconds: float
    entities: list[Entity]
    true_positive: list[str]
    false_positive: list[str]
    false_negative: list[str]
    type_errors: list[str]
    precision: float
    recall: float
    f1: float


@dataclass
class ExtractionResult:
    method: str
    seconds: float
    entities: list[Entity]


def run_benchmark(
    text: str,
    gold: list[Entity],
    methods: list[str],
    gliner_model: str = "urchade/gliner_small-v2.1",
    threshold: float = 0.5,
    gliner_chunk_size: int = DEFAULT_GLINER_CHUNK_SIZE,
    gliner_chunk_overlap: int = DEFAULT_GLINER_CHUNK_OVERLAP,
) -> list[MethodResult]:
    runners: dict[str, Callable[[], list[Entity]]] = {
        "spacy": lambda: extract_spacy(text, include_propn=False),
        "spacy-propn": lambda: extract_spacy(text, include_propn=True),
        "gliner": lambda: extract_gliner(
            text,
            gliner_model,
            threshold,
            gliner_chunk_size,
            gliner_chunk_overlap,
        ),
    }
    results = []
    for method in methods:
        if method not in runners:
            raise ValueError(f"未知方法：{method}")
        started = time.perf_counter()
        predicted = unique_entities(runners[method]())
        results.append(
            score(method, time.perf_counter() - started, predicted, gold)
        )
    return results


def run_extraction(
    text: str,
    methods: list[str],
    gliner_model: str = "urchade/gliner_small-v2.1",
    threshold: float = 0.5,
    gliner_chunk_size: int = DEFAULT_GLINER_CHUNK_SIZE,
    gliner_chunk_overlap: int = DEFAULT_GLINER_CHUNK_OVERLAP,
) -> list[ExtractionResult]:
    runners: dict[str, Callable[[], list[Entity]]] = {
        "spacy": lambda: extract_spacy(text, include_propn=False),
        "spacy-propn": lambda: extract_spacy(text, include_propn=True),
        "gliner": lambda: extract_gliner(
            text,
            gliner_model,
            threshold,
            gliner_chunk_size,
            gliner_chunk_overlap,
        ),
    }
    results = []
    for method in methods:
        if method not in runners:
            raise ValueError(f"未知方法：{method}")
        started = time.perf_counter()
        entities = unique_entities(runners[method]())
        results.append(
            ExtractionResult(method, time.perf_counter() - started, entities)
        )
    return results


def compare_extractions(results: list[ExtractionResult]) -> list[dict]:
    comparisons = []
    for left_index, left in enumerate(results):
        for right in results[left_index + 1 :]:
            left_names = {name_key(item) for item in left.entities}
            right_names = {name_key(item) for item in right.entities}
            comparisons.append(
                {
                    "left": left.method,
                    "right": right.method,
                    "common": sorted(left_names & right_names),
                    "left_only": sorted(left_names - right_names),
                    "right_only": sorted(right_names - left_names),
                }
            )
    return comparisons


def extract_spacy(text: str, include_propn: bool) -> list[Entity]:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [
        Entity(clean(entity.text), SPACY_LABELS[entity.label_])
        for entity in doc.ents
        if entity.label_ in SPACY_LABELS
        and not is_heading(clean(entity.text))
    ]
    if include_propn:
        entities.extend(
            Entity(clean(span.text), "其他專名")
            for span in _proper_noun_spans(doc)
            if not is_heading(clean(span.text))
        )
    return unique_entities(entities)


def extract_gliner(
    text: str,
    model_name: str,
    threshold: float,
    chunk_size: int = DEFAULT_GLINER_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_GLINER_CHUNK_OVERLAP,
) -> list[Entity]:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            "尚未安裝 GLiNER，請執行：python -m pip install gliner"
        ) from exc
    model = GLiNER.from_pretrained(model_name)
    return predict_gliner_chunks(
        model, text, threshold, chunk_size, chunk_overlap
    )


def predict_gliner_chunks(
    model,
    text: str,
    threshold: float,
    chunk_size: int = DEFAULT_GLINER_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_GLINER_CHUNK_OVERLAP,
) -> list[Entity]:
    predictions = []
    for chunk in split_text_chunks(text, chunk_size, chunk_overlap):
        predictions.extend(
            model.predict_entities(
                chunk, list(LABELS), threshold=threshold
            )
        )
    return unique_entities(
        [
            Entity(clean(item["text"]), LABELS[item["label"]])
            for item in predictions
            if item["label"] in LABELS and not is_heading(clean(item["text"]))
        ]
    )


def split_text_chunks(
    text: str,
    chunk_size: int = DEFAULT_GLINER_CHUNK_SIZE,
    overlap: int = DEFAULT_GLINER_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size < 1:
        raise ValueError("GLiNER 分段長度必須大於 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("GLiNER 重疊長度必須介於 0 與分段長度之間")

    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunks.append(text[tokens[start].start() : tokens[end - 1].end()])
        if end == len(tokens):
            break
    return chunks


def _proper_noun_spans(doc):
    start = None
    for index, token in enumerate(doc):
        if token.pos_ == "PROPN":
            start = index if start is None else start
        elif start is not None:
            yield doc[start:index]
            start = None
    if start is not None:
        yield doc[start : len(doc)]


def score(
    method: str,
    seconds: float,
    predicted: list[Entity],
    gold: list[Entity],
) -> MethodResult:
    predicted_map = {name_key(item): item.type for item in predicted}
    gold_map = {name_key(item): item.type for item in gold}
    predicted_keys = set(predicted_map)
    gold_keys = set(gold_map)
    tp = sorted(predicted_keys & gold_keys)
    fp = sorted(predicted_keys - gold_keys)
    fn = sorted(gold_keys - predicted_keys)
    type_errors = [
        f"{key}: 預測 {predicted_map[key]}，正確 {gold_map[key]}"
        for key in sorted(predicted_keys & gold_keys)
        if predicted_map[key] != gold_map[key]
    ]
    precision = len(tp) / (len(tp) + len(fp)) if tp or fp else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if tp or fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return MethodResult(
        method,
        seconds,
        predicted,
        tp,
        fp,
        fn,
        type_errors,
        precision,
        recall,
        f1,
    )


def load_gold(path: str | Path) -> list[Entity]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return unique_entities([Entity(item["text"], item["type"]) for item in payload])


def save_report(path: str | Path, results: list[MethodResult]) -> None:
    payload = []
    for result in results:
        item = asdict(result)
        item["entities"] = [asdict(entity) for entity in result.entities]
        payload.append(item)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_comparison_report(
    path: str | Path, results: list[ExtractionResult]
) -> None:
    payload = {
        "methods": [
            {
                "method": result.method,
                "seconds": result.seconds,
                "entities": [asdict(entity) for entity in result.entities],
            }
            for result in results
        ],
        "comparisons": compare_extractions(results),
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def unique_entities(entities: list[Entity]) -> list[Entity]:
    values = {}
    for entity in entities:
        text = clean(entity.text)
        if text:
            values[(text.casefold(), entity.type)] = Entity(text, entity.type)
    return list(values.values())


def name_key(entity: Entity) -> str:
    return clean(entity.text).casefold()


def clean(value: str) -> str:
    return " ".join(value.split()).strip(" \t\r\n.,;:!?\"“”'()[]")


def is_heading(value: str) -> bool:
    return bool(CHAPTER_PATTERN.match(value))
