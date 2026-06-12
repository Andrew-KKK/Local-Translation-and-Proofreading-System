import argparse
from pathlib import Path

from novel_translator.gutenberg import (
    download_text,
    strip_gutenberg_boilerplate,
    suggested_filename,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="下載 Project Gutenberg 純文字")
    parser.add_argument("url", help="Plain Text UTF-8 網址")
    parser.add_argument("-o", "--output", help="輸出檔案路徑")
    parser.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help="保留 Project Gutenberg 頁首與授權頁尾",
    )
    args = parser.parse_args()

    text = download_text(args.url)
    if not args.keep_boilerplate:
        text = strip_gutenberg_boilerplate(text)
    output = Path(args.output or Path("test-data") / suggested_filename(args.url))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"已儲存：{output.resolve()}")


if __name__ == "__main__":
    main()
