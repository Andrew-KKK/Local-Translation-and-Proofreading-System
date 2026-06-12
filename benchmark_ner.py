import argparse
from pathlib import Path

from novel_translator.ner_benchmark import load_gold, run_benchmark, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="比較小說 NER 方法")
    parser.add_argument("text", help="UTF-8 英文測試文本")
    parser.add_argument("gold", help="人工標註 JSON")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["spacy", "spacy-propn", "gliner"],
        default=["spacy", "spacy-propn"],
    )
    parser.add_argument(
        "--gliner-model", default="urchade/gliner_small-v2.1"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("-o", "--output", default="ner-report.json")
    args = parser.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    results = run_benchmark(
        text,
        load_gold(args.gold),
        args.methods,
        args.gliner_model,
        args.threshold,
    )
    save_report(args.output, results)
    print("method          precision  recall  f1     seconds")
    for result in results:
        print(
            f"{result.method:<15} {result.precision:>9.3f} "
            f"{result.recall:>7.3f} {result.f1:>6.3f} "
            f"{result.seconds:>8.2f}"
        )
        print(f"  漏抓: {', '.join(result.false_negative) or '無'}")
        print(f"  誤抓: {', '.join(result.false_positive) or '無'}")
        print(f"  類型錯誤: {', '.join(result.type_errors) or '無'}")
    print(f"完整報告：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
