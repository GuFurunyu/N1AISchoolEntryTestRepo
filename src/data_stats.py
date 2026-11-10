"""Inspect token counts in the prepared bilingual corpus.

The data preparation script writes English documents followed by Chinese
documents. The first CJK-containing validation line marks the boundary.
"""

import argparse
import re

from .tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="data/processed/val.txt")
    parser.add_argument("--tokenizer", default="tokenizer/spm.model")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    counts = {"english": 0, "chinese": 0}
    lines = {"english": 0, "chinese": 0}
    language = "english"
    with open(args.text, encoding="utf-8") as handle:
        for line in handle:
            if language == "english" and re.search("[\u3400-\u9fff]", line):
                language = "chinese"
            counts[language] += len(tokenizer.encode(line.strip(), out_type=int)) + 1
            lines[language] += 1
    total = sum(counts.values())
    print(f"tokens={counts}")
    print(f"documents={lines}")
    print(f"english_share={counts['english'] / total:.3f}")
    print(f"chinese_share={counts['chinese'] / total:.3f}")


if __name__ == "__main__":
    main()
