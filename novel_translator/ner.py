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


def extract_candidates(text: str) -> list[EntityCandidate]:
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


def _clean(value: str) -> str:
    return " ".join(value.split()).strip(" \t\r\n.,;:!?\"“”'()[]")


def _is_chapter_heading(value: str) -> bool:
    return bool(CHAPTER_PATTERN.match(value))
