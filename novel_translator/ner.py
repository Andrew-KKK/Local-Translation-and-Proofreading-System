from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re


SPACY_LABELS = {
    "PERSON": "人物",
    "GPE": "地名",
    "LOC": "地名",
    "ORG": "組織",
    "PRODUCT": "物件",
    "EVENT": "其他專名",
    "WORK_OF_ART": "其他專名",
}
GLINER_LABELS = {
    "person": "人物",
    "location": "地名",
    "organization": "組織",
    "fictional object": "物件",
    "ability": "能力",
    "title": "稱謂",
    "other proper noun": "其他專名",
}
DEFAULT_GLINER_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_GLINER_CHUNK_SIZE = 220
DEFAULT_GLINER_CHUNK_OVERLAP = 40
TITLE_PATTERN = re.compile(
    r"\b(?:Uncle|Aunt|Mr|Mrs|Miss|Ms|Dr|Doctor|Professor|Captain|King|Queen|"
    r"Prince|Princess|Lord|Lady|Sir)\s+[A-Z][A-Za-z'-]+"
    r"(?:\s+[A-Z][A-Za-z'-]+)*"
)
CHAPTER_PATTERN = re.compile(
    r"^(?:chapter|book|part)\s+(?:\d+|[ivxlcdm]+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EntityCandidate:
    text: str
    suggested_type: str
    source: str


def extract_candidates(
    text: str,
    engine: str = "gliner",
) -> list[EntityCandidate]:
    if engine == "gliner":
        candidates = _extract_gliner_candidates(text)
    elif engine == "spacy":
        candidates = _extract_spacy_candidates(text)
    else:
        raise ValueError(f"不支援的 NER 引擎：{engine}")
    return _apply_title_rules(text, candidates)


def _extract_spacy_candidates(text: str) -> dict[str, EntityCandidate]:
    candidates: dict[str, EntityCandidate] = {}
    doc = _load_spacy_model()(text)
    for entity in doc.ents:
        value = _clean(entity.text)
        if entity.label_ in SPACY_LABELS and not _is_chapter_heading(value):
            candidates[value.casefold()] = EntityCandidate(
                value,
                SPACY_LABELS[entity.label_],
                f"spacy:{entity.label_}",
            )
    return candidates


def _extract_gliner_candidates(text: str) -> dict[str, EntityCandidate]:
    candidates: dict[str, EntityCandidate] = {}
    model = _load_gliner_model()
    for chunk in _split_text_chunks(
        text, DEFAULT_GLINER_CHUNK_SIZE, DEFAULT_GLINER_CHUNK_OVERLAP
    ):
        predictions = model.predict_entities(
            chunk, list(GLINER_LABELS), threshold=0.5
        )
        for prediction in predictions:
            label = prediction.get("label")
            value = _clean(prediction.get("text", ""))
            if (
                label in GLINER_LABELS
                and value
                and not _is_chapter_heading(value)
            ):
                candidates.setdefault(
                    value.casefold(),
                    EntityCandidate(
                        value,
                        GLINER_LABELS[label],
                        f"gliner:{label}",
                    ),
                )
    return candidates


def _apply_title_rules(
    text: str,
    candidates: dict[str, EntityCandidate],
) -> list[EntityCandidate]:
    title_candidates = []
    for match in TITLE_PATTERN.finditer(text):
        value = _clean(match.group())
        title_candidates.append(value)
    for title in title_candidates:
        title_words = {word.casefold() for word in title.split()}
        for key, candidate in list(candidates.items()):
            candidate_words = {word.casefold() for word in candidate.text.split()}
            if candidate_words < title_words:
                del candidates[key]
        candidates[title.casefold()] = EntityCandidate(title, "稱謂", "title-rule")
    return list(candidates.values())


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "缺少 spaCy 英文模型，請執行："
            "python -m spacy download en_core_web_sm"
        ) from exc


@lru_cache(maxsize=1)
def _load_gliner_model():
    try:
        from gliner import GLiNER
        return GLiNER.from_pretrained(DEFAULT_GLINER_MODEL)
    except ImportError as exc:
        raise RuntimeError(
            "缺少 GLiNER，請執行：python -m pip install gliner"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "GLiNER／PyTorch 無法啟動，請確認 CPU 版 PyTorch 與 "
            "Microsoft Visual C++ Runtime 已正確安裝"
        ) from exc


def _split_text_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
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


def _clean(value: str) -> str:
    return " ".join(value.split()).strip(" \t\r\n.,;:!?\"“”'()[]")


def _is_chapter_heading(value: str) -> bool:
    return bool(CHAPTER_PATTERN.match(value))
