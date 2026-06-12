from dataclasses import dataclass
import json
import math
import re
import time

from .glossary import Term, parse_glossary, persona_section
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
必須翻譯輸入中的全部內容，包括章節標題與正文，不可只翻譯標題或摘要。
只輸出完整譯文，不要解釋、術語表或 Markdown 表格。"""

REVIEW_SYSTEM = """你是繁體中文小說翻譯的品質審查員。對照英文原文、術語表與初稿，
修正漏譯、誤譯、譯名違規、代詞錯置及翻譯腔，不得增添情節。
只輸出修訂後的完整譯文。"""

TERM_TYPES = ["人物", "地名", "組織", "物件", "能力", "稱謂", "其他專名"]
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
}
DEFAULT_TERM_BATCH_SIZE = 8
TERM_CONTEXT_MAX_CHARS = 1000
UNRESOLVED_TARGET = "（待人工確認）"


class TermBatchError(RuntimeError):
    def __init__(
        self,
        batch_number: int,
        batch_count: int,
        partial_terms: list[Term],
        cause: Exception,
    ):
        self.batch_number = batch_number
        self.batch_count = batch_count
        self.partial_terms = partial_terms
        super().__init__(
            f"術語提案第 {batch_number}/{batch_count} 批失敗；"
            f"已保留前 {len(partial_terms)} 筆結果。原因：{cause}"
        )


def build_term_schema(count: int) -> dict:
    properties = {}
    required = []
    for index in range(count):
        key = f"entity_{index}"
        required.append(key)
        properties[key] = {
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
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
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("模型沒有完整回傳所有 NER 候選") from exc
    if not isinstance(payload, dict):
        raise ValueError("模型沒有完整回傳所有 NER 候選")

    terms = []
    for index, candidate in enumerate(candidates):
        value = payload.get(f"entity_{index}", {})
        if not isinstance(value, dict):
            value = {}
        target = value.get("target", "")
        term_type = value.get("type", "")
        remarks = value.get("remarks", "")
        target = target.strip() if isinstance(target, str) else ""
        term_type = term_type.strip() if isinstance(term_type, str) else ""
        remarks = remarks.strip() if isinstance(remarks, str) else ""
        terms.append(
            Term(
                candidate.text,
                target or candidate.text,
                term_type if term_type in TERM_TYPES else candidate.suggested_type,
                chapter,
                remarks,
            )
        )
    return terms


@dataclass
class TranslationPipeline:
    client: object
    model: str = "llama-3-taiwan-8b"
    chunk_chars: int = 3000
    ner_engine: str = "gliner"
    term_batch_size: int = DEFAULT_TERM_BATCH_SIZE
    last_ner_candidates: list[str] | None = None
    last_scan_batch_count: int = 0
    stage_timings: list[tuple[str, float]] | None = None

    def __post_init__(self):
        self.stage_timings = []

    def scan(self, source: str, glossary: str, chapter: str) -> list[Term]:
        if not source.strip():
            raise ValueError("請先提供英文原文")
        existing = parse_glossary(glossary)
        known_keys = {term.source.casefold() for term in existing}
        candidates = [
            item
            for item in extract_candidates(source, self.ner_engine)
            if item.text.casefold() not in known_keys
        ]
        self.last_ner_candidates = [item.text for item in candidates]
        if not candidates:
            self.last_scan_batch_count = 0
            return []
        batches = context_candidate_batches(
            source, candidates, self.term_batch_size
        )
        self.last_scan_batch_count = len(batches)
        completed = []
        for batch_number, (context, batch) in enumerate(batches, 1):
            prompt = build_scan_prompt(
                batch,
                context,
                existing,
                chapter or "未指定",
                batch_number,
                len(batches),
            )
            try:
                raw = self.client.generate(
                    self.model,
                    SCAN_SYSTEM,
                    prompt,
                    format_schema=build_term_schema(len(batch)),
                )
                terms = parse_classified_terms(
                    raw, batch, chapter or "未指定"
                )
                completed.extend(
                    self._repair_untranslated_targets(terms, context)
                )
            except Exception as exc:
                raise TermBatchError(
                    batch_number, len(batches), completed, exc
                ) from exc
        return completed

    def _repair_untranslated_targets(
        self, terms: list[Term], context: str
    ) -> list[Term]:
        result = list(terms)
        schema = {
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
            },
            "required": ["target"],
            "additionalProperties": False,
        }
        for index, term in enumerate(terms):
            if contains_han(term.target):
                continue
            prompt = (
                "請將下列英語專名音譯或意譯為自然的臺灣繁體中文。"
                "target 必須包含中文字，不可留白或照抄英文。\n\n"
                f"專名：{term.source}\n語境：{context}"
            )
            try:
                raw = self.client.generate(
                    self.model,
                    "你是英語小說專名的繁體中文譯名編輯。",
                    prompt,
                    format_schema=schema,
                )
                target = json.loads(raw)["target"]
                if not isinstance(target, str) or not contains_han(target):
                    raise ValueError("修復後的譯名仍不含中文")
                result[index] = Term(
                    term.source,
                    target,
                    term.type,
                    term.first_chapter,
                    term.remarks,
                )
            except Exception:
                note = "模型未產生有效繁中譯名，請人工修改"
                result[index] = Term(
                    term.source,
                    UNRESOLVED_TARGET,
                    term.type,
                    term.first_chapter,
                    f"{term.remarks}；{note}".strip("；"),
                )
        return result

    def translate(self, source: str, glossary: str) -> str:
        if not source.strip():
            raise ValueError("請先提供英文原文")
        chunks = split_text(source, self.chunk_chars)
        outputs = []
        context = translation_context(glossary)
        for index, chunk in enumerate(chunks, 1):
            options, timeout = translation_limits(chunk)
            prompt = (
                f"翻譯規則與術語：\n{context}\n\n"
                f"第 {index}/{len(chunks)} 段英文原文：\n{chunk}"
            )
            raw = self._generate_timed(
                f"初稿 {index}/{len(chunks)}",
                TRANSLATE_SYSTEM,
                prompt,
                TEXT_SCHEMA,
                options,
                timeout,
            )
            translation = parse_translation(raw)
            issues = translation_quality_issues(chunk, translation)
            if issues:
                retry_prompt = (
                    f"上一版譯文不合格：{'；'.join(issues)}。\n"
                    "請重新完整翻譯，不可省略正文，也不可輸出術語表。\n\n"
                    f"翻譯規則與術語：\n{context}\n\n"
                    f"第 {index}/{len(chunks)} 段英文原文：\n{chunk}"
                )
                raw = self._generate_timed(
                    f"重試 {index}/{len(chunks)}",
                    TRANSLATE_SYSTEM,
                    retry_prompt,
                    TEXT_SCHEMA,
                    options,
                    timeout,
                )
                translation = parse_translation(raw)
                issues = translation_quality_issues(chunk, translation)
                if issues:
                    raise ValueError(
                        "模型重試後仍未產生完整譯文：" + "；".join(issues)
                    )
            outputs.append(translation)
        return "\n\n".join(outputs)

    def _generate_timed(
        self,
        stage: str,
        system: str,
        prompt: str,
        format_schema: dict,
        options: dict,
        timeout: int,
    ) -> str:
        started = time.perf_counter()
        try:
            return self.client.generate(
                self.model,
                system,
                prompt,
                format_schema=format_schema,
                options=options,
                timeout=timeout,
            )
        finally:
            self.stage_timings.append(
                (stage, time.perf_counter() - started)
            )

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
        corrected = translation
        source_folded = source.casefold()
        for term in parse_glossary(glossary):
            if term.source.casefold() not in source_folded:
                continue
            corrected = re.sub(
                rf"(?<![A-Za-z]){re.escape(term.source)}(?![A-Za-z])",
                term.target,
                corrected,
                flags=re.IGNORECASE,
            )
        corrected = re.sub(
            r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])",
            "",
            corrected,
        )
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


def chunked(values: list, size: int):
    if size < 1:
        raise ValueError("術語批次大小必須大於 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def split_context_blocks(
    source: str, max_chars: int = TERM_CONTEXT_MAX_CHARS
) -> list[str]:
    blocks = []
    for paragraph in re.split(r"\n\s*\n", source):
        paragraph = " ".join(paragraph.split()).strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            blocks.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    blocks.append(current)
                    current = ""
                blocks.extend(split_long_context(sentence, max_chars))
                continue
            combined = f"{current} {sentence}".strip()
            if current and len(combined) > max_chars:
                blocks.append(current)
                current = sentence
            else:
                current = combined
        if current:
            blocks.append(current)
    return blocks


def split_long_context(
    text: str, max_chars: int, overlap: int = 100
) -> list[str]:
    overlap = min(overlap, max_chars // 3)
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            word_boundary = text.rfind(" ", start + max_chars // 2, end)
            if word_boundary > start:
                end = word_boundary
        blocks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
        next_space = text.find(" ", start)
        if 0 <= next_space < end:
            start = next_space + 1
    return [block for block in blocks if block]


def context_candidate_batches(
    source: str,
    candidates: list,
    batch_size: int,
) -> list[tuple[str, list]]:
    blocks = split_context_blocks(source)
    grouped: list[tuple[str, list]] = [(block, []) for block in blocks]
    unmatched = []
    for candidate in candidates:
        key = candidate.text.casefold()
        matched = False
        for index, (block, values) in enumerate(grouped):
            if key in block.casefold():
                grouped[index][1].append(candidate)
                matched = True
                break
        if not matched:
            unmatched.append(candidate)

    batches = []
    for block, values in grouped:
        for batch in chunked(values, batch_size):
            batches.append((block, batch))
    fallback_context = blocks[0] if blocks else source[:TERM_CONTEXT_MAX_CHARS]
    for batch in chunked(unmatched, batch_size):
        batches.append((fallback_context, batch))
    return batches


def build_scan_prompt(
    candidates: list,
    context: str,
    existing: list[Term],
    chapter: str,
    batch_number: int,
    batch_count: int,
) -> str:
    candidate_lines = [
        f"- entity_{index}: {item.text} | 建議類型：{item.suggested_type}"
        for index, item in enumerate(candidates)
    ]
    combined_context = context.casefold()
    related = [
        term
        for term in existing
        if term.source.casefold() in combined_context
    ]
    known = "\n".join(
        f"- {term.source} → {term.target}" for term in related
    ) or "（此批語境無相關既有術語）"
    return (
        f"章節：{chapter}\n"
        f"批次：{batch_number}/{batch_count}\n\n"
        f"候選實體：\n{chr(10).join(candidate_lines)}\n\n"
        f"共用語境：\n{context}\n\n"
        f"相關既有術語：\n{known}"
    )


def translation_context(glossary: str) -> str:
    terms = parse_glossary(glossary)
    term_lines = [
        f"- {term.source} → {term.target}（{term.type}）"
        for term in terms
    ]
    persona = persona_section(glossary)
    sections = ["指定譯名：\n" + ("\n".join(term_lines) or "（無）")]
    if persona and not persona.startswith("<!--"):
        sections.append("角色語氣與人設：\n" + persona)
    return "\n\n".join(sections)


def translation_limits(source: str) -> tuple[dict, int]:
    word_count = len(
        re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", source)
    )
    output_tokens = max(256, min(2048, math.ceil(word_count * 2.2) + 128))
    timeout = max(120, min(420, math.ceil(60 + output_tokens / 3)))
    return {"num_predict": output_tokens}, timeout


def translation_quality_issues(source: str, translation: str) -> list[str]:
    issues = []
    forbidden = ("小說術語表", "| 原文名稱 |", "## 專有名詞")
    if any(marker in translation for marker in forbidden):
        issues.append("輸出了術語表而非小說譯文")

    source_words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", source)
    han_count = len(re.findall(r"[\u3400-\u9fff]", translation))
    minimum_han = max(2, math.ceil(len(source_words) * 0.35))
    if han_count < minimum_han:
        issues.append(
            f"譯文過短或漏譯正文（中文 {han_count} 字，至少需約 {minimum_han} 字）"
        )

    translated_english = re.findall(
        r"[A-Za-z]+(?:['’-][A-Za-z]+)*", translation
    )
    if source_words and len(translated_english) > max(6, len(source_words) // 2):
        issues.append("譯文保留過多英文原文")
    return issues


def glossary_violations(source: str, translation: str, glossary: str) -> list[str]:
    source_folded = source.casefold()
    return [
        f"{term.source} → {term.target}"
        for term in parse_glossary(glossary)
        if term.source.casefold() in source_folded and term.target not in translation
    ]


def residual_glossary_sources(
    source: str, translation: str, glossary: str
) -> list[str]:
    source_folded = source.casefold()
    return [
        term.source
        for term in parse_glossary(glossary)
        if term.source.casefold() in source_folded
        and re.search(
            rf"(?<![A-Za-z]){re.escape(term.source)}(?![A-Za-z])",
            translation,
            re.IGNORECASE,
        )
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
