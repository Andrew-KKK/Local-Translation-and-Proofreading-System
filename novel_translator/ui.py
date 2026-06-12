from datetime import datetime
from pathlib import Path
import tempfile

import gradio as gr

from .glossary import (
    DEFAULT_GLOSSARY,
    Term,
    merge_terms,
    parse_glossary,
    persona_section,
    render_glossary,
)
from .gutenberg import download_text, strip_gutenberg_boilerplate, suggested_filename
from .ollama_client import OllamaClient
from .pipeline import TranslationPipeline, glossary_violations


HEADERS = ["原文", "繁中譯名", "類型", "首次章節", "備註"]
PREFERRED_MODEL = "hf.co/chienweichang/Llama-3-Taiwan-8B-Instruct-GGUF:Q4_K_M"


def pipeline(model: str, url: str, chunk_chars: float) -> TranslationPipeline:
    return TranslationPipeline(
        OllamaClient(url), model.strip() or PREFERRED_MODEL, int(chunk_chars)
    )


def installed_models(url: str = "http://localhost:11434") -> list[str]:
    try:
        return OllamaClient(url).list_models()
    except Exception:
        return []


def load_text(path: str | None, current: str) -> str:
    return Path(path).read_text(encoding="utf-8") if path else current


def download_test_text(url: str):
    try:
        text = strip_gutenberg_boilerplate(download_text(url))
        folder = Path(tempfile.mkdtemp(prefix="gutenberg-"))
        output = folder / suggested_filename(url)
        output.write_text(text, encoding="utf-8")
        return str(output), f"測試文本下載完成，共 {len(text):,} 個字元。"
    except Exception as exc:
        return None, f"下載失敗：{exc}"


def scan_terms(source, glossary, chapter, model, url, chunk_chars):
    try:
        terms = pipeline(model, url, chunk_chars).scan(source, glossary, chapter)
        rows = [
            [t.source, t.target, t.type, t.first_chapter, t.remarks] for t in terms
        ]
        return rows, f"找到 {len(rows)} 筆候選，請刪除不採用的列後批准。"
    except Exception as exc:
        return [], f"掃描失敗：{exc}"


def approve_terms(rows, glossary):
    try:
        keys = ["source", "target", "type", "first_chapter", "remarks"]
        approved = [
            Term.from_dict(dict(zip(keys, (str(cell or "") for cell in row))))
            for row in rows
        ]
        merged = merge_terms(parse_glossary(glossary), approved)
        updated = render_glossary(merged, persona_section(glossary))
        return updated, f"已批准 {len(approved)} 筆術語。"
    except Exception as exc:
        return glossary, f"批准失敗：{exc}"


def translate(source, glossary, model, url, chunk_chars):
    try:
        flow = pipeline(model, url, chunk_chars)
        result = flow.translate(source, glossary)
        result, violations = flow.enforce_glossary(source, result, glossary)
        message = "翻譯初稿完成，並已自動檢查指定譯名。"
        if violations:
            message += " 自動修正後仍有違規：" + "、".join(violations)
        return result, message
    except Exception as exc:
        return "", f"翻譯失敗：{exc}"


def review(source, draft, glossary, model, url, chunk_chars):
    try:
        flow = pipeline(model, url, chunk_chars)
        result = flow.review(source, draft, glossary)
        result, violations = flow.enforce_glossary(source, result, glossary)
        message = "品質審查完成，並已自動修正偵測到的譯名違規。"
        if violations:
            message += " 仍有譯名違規：" + "、".join(violations)
        return result, message
    except Exception as exc:
        return draft, f"品質審查失敗：{exc}"


def export_files(translation: str, glossary: str):
    if not translation.strip():
        raise gr.Error("尚無譯文可匯出")
    folder = Path(tempfile.mkdtemp(prefix="novel-translator-"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = folder / f"translation-{stamp}.md"
    terms = folder / f"glossary-{stamp}.md"
    output.write_text(translation, encoding="utf-8")
    terms.write_text(glossary, encoding="utf-8")
    return str(output), str(terms)


def build_app() -> gr.Blocks:
    models = installed_models()
    default_model = (
        PREFERRED_MODEL
        if PREFERRED_MODEL in models
        else (models[0] if models else PREFERRED_MODEL)
    )
    with gr.Blocks(title="本地端小說翻譯協作系統") as app:
        gr.Markdown(
            "# 本地端小說翻譯協作系統\n"
            "英翻繁中原型。術語必須由使用者批准後才會加入 Markdown。"
        )
        status = gr.Markdown("準備就緒。")
        with gr.Accordion("下載 Project Gutenberg 測試文本", open=False):
            gutenberg_url = gr.Textbox(
                placeholder="https://www.gutenberg.org/cache/epub/55/pg55.txt",
                label="Plain Text UTF-8 網址",
            )
            download_button = gr.Button("下載並移除 Gutenberg 頁首頁尾")
            test_text_download = gr.File(label="下載清理後測試文本")
        with gr.Accordion("模型設定", open=False):
            model = gr.Dropdown(
                choices=models,
                value=default_model,
                allow_custom_value=True,
                label="Ollama 模型",
            )
            url = gr.Textbox(value="http://localhost:11434", label="Ollama 網址")
            chunk_chars = gr.Slider(
                1000, 6000, value=3000, step=250, label="每段最大字元數"
            )
        source_file = gr.File(
            label="上傳 UTF-8 英文 .txt 或 .md", type="filepath"
        )
        chapter = gr.Textbox(value="Chapter 1", label="章節名稱")
        source = gr.Textbox(lines=14, label="英文原文")
        source_file.change(load_text, [source_file, source], source)
        download_button.click(
            download_test_text,
            gutenberg_url,
            [test_text_download, status],
        )

        gr.Markdown("## 1. 掃描與批准術語")
        glossary_file = gr.File(
            label="載入既有 glossary.md", file_types=[".md"], type="filepath"
        )
        glossary = gr.Textbox(value=DEFAULT_GLOSSARY, lines=14, label="glossary.md")
        glossary_file.change(load_text, [glossary_file, glossary], glossary)
        scan_button = gr.Button("掃描術語候選", variant="primary")
        scan_status = gr.Markdown("尚未掃描。")
        candidates = gr.Dataframe(
            headers=HEADERS, datatype=["str"] * 5, type="array", label="術語候選"
        )
        approve_button = gr.Button("批准表格中的術語")

        gr.Markdown("## 2. 翻譯與品質審查")
        translate_button = gr.Button("產生翻譯初稿", variant="primary")
        draft = gr.Textbox(lines=14, label="翻譯初稿")
        review_button = gr.Button("執行品質審查")
        reviewed = gr.Textbox(lines=14, label="審查後譯文")

        gr.Markdown("## 3. 匯出")
        export_button = gr.Button("匯出審查稿與 glossary.md")
        translation_download = gr.File(label="下載譯文")
        glossary_download = gr.File(label="下載術語表")

        scan_button.click(
            scan_terms,
            [source, glossary, chapter, model, url, chunk_chars],
            [candidates, scan_status],
        )
        approve_button.click(
            approve_terms, [candidates, glossary], [glossary, status]
        )
        translate_button.click(
            translate,
            [source, glossary, model, url, chunk_chars],
            [draft, status],
        )
        review_button.click(
            review,
            [source, draft, glossary, model, url, chunk_chars],
            [reviewed, status],
        )
        export_button.click(
            export_files,
            [reviewed, glossary],
            [translation_download, glossary_download],
        )
    return app
