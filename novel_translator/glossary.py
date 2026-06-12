from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re


HEADER = "| 原文名稱 | 中文譯名 | 類型 | 首次出現章節 | 備註 |"
DIVIDER = "| --- | --- | --- | --- | --- |"
DEFAULT_GLOSSARY = f"""# 小說術語表

## 專有名詞
{HEADER}
{DIVIDER}

## 角色語氣與人設
<!-- 可在此加入人物關係、固定稱呼與語氣規範。 -->
"""


@dataclass(frozen=True)
class Term:
    source: str
    target: str
    type: str
    first_chapter: str = ""
    remarks: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Term":
        fields = ("source", "target", "type", "first_chapter", "remarks")
        if any(not isinstance(data.get(key, ""), str) for key in fields):
            raise ValueError("術語欄位必須是字串")
        values = [clean(data.get(key, "")) for key in fields]
        if not all(values[:3]):
            raise ValueError("source、target 與 type 不可空白")
        return cls(*values)


def clean(value: str) -> str:
    return " ".join(value.replace("|", "／").split())


def parse_glossary(markdown: str) -> list[Term]:
    terms = []
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.strip().startswith("|") and len(cells) == 5:
            if cells[0] not in {"原文名稱", "---"}:
                terms.append(Term(*cells))
    return terms


def parse_proposals(raw: str) -> list[Term]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    text = fenced.group(1) if fenced else raw
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("模型沒有回傳有效 JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("術語提案必須是 JSON 陣列")
    return [Term.from_dict(item) for item in payload]


def merge_terms(existing: list[Term], approved: list[Term]) -> list[Term]:
    result = list(existing)
    positions = {term.source.casefold(): i for i, term in enumerate(result)}
    for term in approved:
        key = term.source.casefold()
        if key in positions:
            result[positions[key]] = term
        else:
            positions[key] = len(result)
            result.append(term)
    return result


def persona_section(markdown: str) -> str:
    match = re.search(r"## 角色語氣與人設\s*(.*)\Z", markdown, re.DOTALL)
    return match.group(1).strip() if match else ""


def render_glossary(terms: list[Term], persona: str = "") -> str:
    rows = "\n".join(
        f"| {t.source} | {t.target} | {t.type} | {t.first_chapter} | {t.remarks} |"
        for t in terms
    )
    note = persona or "<!-- 可加入人物關係、固定稱呼與語氣規範。 -->"
    return (
        f"# 小說術語表\n\n## 專有名詞\n{HEADER}\n{DIVIDER}\n{rows}\n\n"
        f"## 角色語氣與人設\n{note}\n"
    )
