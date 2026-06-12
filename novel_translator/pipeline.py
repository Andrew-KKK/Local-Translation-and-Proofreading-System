from dataclasses import dataclass
import json
import re

from .glossary import Term, parse_glossary
from .ner import extract_candidates


SCAN_SYSTEM = """你是英語小說的專有名詞編輯。程式已從原文抽出候選實體。

只允許以下類型：人物、地名、組織、物件、能力、稱謂、其他專名。
具名動物視為人物角色，不可分類為物件。
每個 entity_N 都必須輸出，不能省略、合併或新增候選。
target 必須是含中文字的臺灣繁體中文譯名，不可照抄英文。
只需填入譯名、類型與簡短備註。
例如 Dorothy 的 target 應為「桃樂絲」，Kansas 應為「堪薩斯州」。"""

TRANSLATE_SYSTEM = """你是英翻繁中的小說譯者。忠實保留資訊、敘事視角、段落與對話。
使用自然的臺灣繁體中文。術語表中的譯名、稱呼與人物語氣是硬性規則。
只輸出完整譯文，不要解釋。"""

REVIEW_SYSTEM = """你是繁體中文小說翻譯的品質審查員。對照英文原文、術語表與初稿，
修正漏譯、誤譯、譯名違規、代詞錯置及翻譯腔，不得增添情節。
只輸出修訂後的完整譯文。"""

TERM_FIX_SYSTEM = """你是繁體中文小說的術語修訂員。
只修正指定的譯名違規，並視需要調整附近語序使中文自然。
不得改變情節、刪減資訊、加入解釋，或改動未列出的專有名詞。"""

TERM_TYPES = ["人物", "地名", "組織", "物件", "能力", "稱謂", "其他專名"]
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
}


def build_term_schema(count: int) -> dict:
    properties = {}
    required = []
    for index in range(count):
        key = f"entity_{index}"
        required.append(key)
        properties[key] = {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "type": {"type": "string", "enum": TERM_TYPES},
                "remarks": {"type": "string"},
            },
            "required": ["target", "type", "remarks"],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def parse_classified_terms(raw: str, candidates: list, chapter: str) -> list[Term]:
    try:
        payload = json.loads(raw)
        terms = []
        for index, candidate in enumerate(candidates):
            value = payload[f"entity_{index}"]
            terms.append(
                Term.from_dict(
                    {
                        "source": candidate.text,
                        "target": value["target"],
                        "type": value["type"],
                        "first_chapter": chapter,
                        "remarks": value["remarks"],
                    }
                )
            )
        return terms
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("模型沒有完整回傳所有 spaCy 候選") from exc


@dataclass
class TranslationPipeline:
    client: object
    model: str = "llama-3-taiwan-8b"
    chunk_chars: int = 3000

    def scan(self, source: str, glossary: str, chapter: str) -> list[Term]:
        if not source.strip():
            raise ValueError("請先提供英文原文")
        existing = parse_glossary(glossary)
        known_keys = {term.source.casefold() for term in existing}
        candidates = [
            item
            for item in extract_candidates(source)
            if item.text.casefold() not in known_keys
        ]
        if not candidates:
            return []
        known = "\n".join(
            f"- {term.source} -> {term.target}" for term in existing
        ) or "（無）"
        candidate_text = "\n".join(
            f"- entity_{index}: {item.text} | 建議類型：{item.suggested_type}"
            for index, item in enumerate(candidates)
        )
        prompt = (
            f"章節：{chapter or '未指定'}\n\n候選實體：\n{candidate_text}\n\n"
            f"既有術語：\n{known}\n\n語境原文：\n{source}"
        )
        raw = self.client.generate(
            self.model,
            SCAN_SYSTEM,
            prompt,
            format_schema=build_term_schema(len(candidates)),
        )
        terms = parse_classified_terms(raw, candidates, chapter or "未指定")
        return self._repair_untranslated_targets(terms, source)

    def _repair_untranslated_targets(
        self, terms: list[Term], source: str
    ) -> list[Term]:
        missing = [
            (index, term)
            for index, term in enumerate(terms)
            if not contains_han(term.target)
        ]
        if not missing:
            return terms
        properties = {
            f"target_{index}": {"type": "string"}
            for index, _ in missing
        }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        names = "\n".join(
            f"- target_{index}: {term.source}" for index, term in missing
        )
        prompt = (
            "請將下列英語專名音譯或意譯為自然的臺灣繁體中文。"
            "每個值必須包含中文字，不可照抄英文：\n"
            f"{names}\n\n語境：\n{source}"
        )
        raw = self.client.generate(
            self.model,
            "你是英語小說專名的繁體中文譯名編輯。",
            prompt,
            format_schema=schema,
        )
        try:
            repaired = json.loads(raw)
            result = list(terms)
            for index, term in missing:
                target = repaired[f"target_{index}"]
                result[index] = Term(
                    term.source,
                    target,
                    term.type,
                    term.first_chapter,
                    term.remarks,
                )
            return result
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("模型未能補齊繁中術語譯名") from exc

    def translate(self, source: str, glossary: str) -> str:
        if not source.strip():
            raise ValueError("請先提供英文原文")
        chunks = split_text(source, self.chunk_chars)
        outputs = []
        for index, chunk in enumerate(chunks, 1):
            prompt = (
                f"術語表：\n{glossary}\n\n"
                f"第 {index}/{len(chunks)} 段英文原文：\n{chunk}"
            )
            raw = self.client.generate(
                self.model, TRANSLATE_SYSTEM, prompt, format_schema=TEXT_SCHEMA
            )
            outputs.append(parse_translation(raw))
        return "\n\n".join(outputs)

    def review(self, source: str, draft: str, glossary: str) -> str:
        if not draft.strip():
            raise ValueError("尚無翻譯初稿可供審查")
        prompt = (
            f"術語表：\n{glossary}\n\n英文原文：\n{source}\n\n"
            f"繁中初稿：\n{draft}"
        )
        raw = self.client.generate(
            self.model, REVIEW_SYSTEM, prompt, format_schema=TEXT_SCHEMA
        )
        return parse_translation(raw)

    def enforce_glossary(
        self, source: str, translation: str, glossary: str
    ) -> tuple[str, list[str]]:
        violations = glossary_violations(source, translation, glossary)
        if not violations:
            return translation, []
        prompt = (
            "必須修正的譯名如下：\n- "
            + "\n- ".join(violations)
            + f"\n\n英文原文：\n{source}\n\n目前譯文：\n{translation}"
        )
        raw = self.client.generate(
            self.model, TERM_FIX_SYSTEM, prompt, format_schema=TEXT_SCHEMA
        )
        corrected = parse_translation(raw)
        return corrected, glossary_violations(source, corrected, glossary)


def parse_translation(raw: str) -> str:
    try:
        value = json.loads(raw)
        translation = value["translation"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("模型沒有回傳有效的翻譯資料") from exc
    if not isinstance(translation, str) or not translation.strip():
        raise ValueError("模型回傳了空白譯文")
    return translation.strip()


def contains_han(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def glossary_violations(source: str, translation: str, glossary: str) -> list[str]:
    source_folded = source.casefold()
    return [
        f"{term.source} → {term.target}"
        for term in parse_glossary(glossary)
        if term.source.casefold() in source_folded and term.target not in translation
    ]


def split_text(text: str, max_chars: int) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks, current = [], ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        pieces = [
            paragraph[i : i + max_chars]
            for i in range(0, len(paragraph), max_chars)
        ]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks
