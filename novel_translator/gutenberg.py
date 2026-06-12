from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


START_MARKER = re.compile(
    r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}",
    re.IGNORECASE,
)
END_MARKER = re.compile(
    r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}",
    re.IGNORECASE,
)


def download_text(url: str, timeout: int = 60) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("請輸入有效的 HTTP 或 HTTPS 網址")
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    try:
        response = session.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "LocalNovelTranslator/0.1 "
                    "(educational Project Gutenberg text downloader)"
                )
            },
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"下載失敗：{exc}") from exc
    try:
        return response.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return response.text.lstrip("\ufeff")


def strip_gutenberg_boilerplate(text: str) -> str:
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    begin = start.end() if start else 0
    finish = end.start() if end else len(text)
    if finish <= begin:
        raise ValueError("無法辨識 Project Gutenberg 文字範圍")
    cleaned = text[begin:finish].strip()
    return re.sub(r"\r\n?", "\n", cleaned) + "\n"


def suggested_filename(url: str) -> str:
    name = Path(urlparse(url).path).name or "gutenberg-book.txt"
    return name if name.lower().endswith(".txt") else f"{name}.txt"
