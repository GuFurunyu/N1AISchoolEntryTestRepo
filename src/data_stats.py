"""统计准备好的文本文件中，中英文各自编码成多少 token。

prepare_data.py 先写英文文档，再写中文文档。本脚本用第一行出现的
CJK 汉字作为语言切换的近似标志；若英文来源里混入汉字，这个判断会提前切换。
它只是审计工具，不参与训练循环。
"""

import argparse
import re

from .tokenizer import load_tokenizer


def main():
    """逐行重新编码文本，输出每种语言的文档数、token 数和占比。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="data/processed/val.txt")
    parser.add_argument("--tokenizer", default="tokenizer/spm.model")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    # counts 记录 token 数；lines 记录文件中的文档行数。
    counts = {"english": 0, "chinese": 0}
    lines = {"english": 0, "chinese": 0}
    # 文件按英文在前、中文在后组织，所以从英文开始，切到中文后不切回。
    language = "english"
    with open(args.text, encoding="utf-8") as handle:
        for line in handle:
            # \u3400-\u9fff 是常见 CJK 汉字区间，用于推断中文段开始。
            if language == "english" and re.search("[\u3400-\u9fff]", line):
                language = "chinese"
            # 与数据编码一致，每篇文档另加一个 EOS token，因此计数 +1。
            counts[language] += len(tokenizer.encode(line.strip(), out_type=int)) + 1
            lines[language] += 1
    # 这里的比例按 token 计算，而非按字符或文档篇数计算。
    total = sum(counts.values())
    print(f"tokens={counts}")
    print(f"documents={lines}")
    print(f"english_share={counts['english'] / total:.3f}")
    print(f"chinese_share={counts['chinese'] / total:.3f}")


if __name__ == "__main__":
    main()
