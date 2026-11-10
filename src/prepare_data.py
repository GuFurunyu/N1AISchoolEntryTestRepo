"""把网上的原始文档变成模型可读的连续 token 文件。

执行顺序：下载/读取数据集 -> 清洗并划分 train/val 文本 ->
只用 train.txt 训练 tokenizer -> 把两份文本编码为 uint16 的 .bin 文件。
使用 --skip-download 时，直接读取已经存在的 train.txt 和 val.txt。
"""

import argparse
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .tokenizer import encode_text, train_sentencepiece, load_tokenizer


def _write_documents(documents: Iterable[str], output: Path, max_chars: int) -> int:
    """把文档逐行写入文本文件，并返回实际写入的文档数。

这是早期保留的辅助函数；当前主流程 build_text_splits() 没有调用它。
它限制的是 Python 字符数，不是 UTF-8 字节数，也不是 tokenizer 的 token 数。
"""
    # output.parent 是文件所在文件夹；parents=True 可连同上级目录一起创建。
    output.parent.mkdir(parents=True, exist_ok=True)
    # count 记录文档篇数；written 记录写入的文本字符数和换行符数。
    count = 0
    written = 0
    # "w" 会覆盖旧文件；with 结束时自动关闭文件。
    with output.open("w", encoding="utf-8") as handle:
        for document in documents:
            # split() 按任意空白切开，join() 用单个空格连接，
            # 因此连续空格、制表符和原来的换行/段落都会被压成空格。
            text = " ".join(str(document).split())
            # 跳过清洗后过短的内容，不计入写入数量。
            if len(text) < 20:
                continue
            # 下一篇若超过字符预算，就停止整个循环，后面的短文档也不会再尝试。
            # 这里未预先计入即将写入的换行符，实际字符数可能比 max_chars 多 1。
            if written + len(text) > max_chars:
                break
            # 每篇文档占文件中的一行；加一个换行符供下次按行读取。
            handle.write(text + "\n")
            written += len(text) + 1
            count += 1
    return count


def _load_documents(dataset_name: str, config: str, split: str, cache_dir: str):
    """从 Hugging Face 数据集逐篇产出文档正文。

这是生成器函数：yield 一篇后才继续取下一篇，调用方不必一次保存
所有正文。datasets 库本身仍可能下载并建立本地缓存。
"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before downloading the corpus") from exc
    # Wikipedia 需要指定 20231101.zh 这样的配置；TinyStories 使用默认配置。
    if config:
        dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    # 两个来源的正文列通常叫 text；如果没有，尝试 content。
    text_key = "text" if "text" in dataset.column_names else "content"
    for row in dataset:
        value = row.get(text_key, "")
        # 空正文不交给后面的清洗过程。
        if value:
            yield value


def build_text_splits(args) -> Tuple[Path, Path]:
    """下载/读取两种语言，按文档抽出验证集，写 train.txt 和 val.txt。

返回两个文本文件路径。语言按“先英文、再中文”的顺序写入，后续代码
利用这个顺序和实测 token 边界分别采样及评估两种语言。
"""
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    cache_dir = Path(args.cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 此时只创建生成器；真正访问数据集发生在下面遍历时。
    english = _load_documents("roneneldan/TinyStories", None, "train", str(cache_dir))
    chinese = _load_documents("wikimedia/wikipedia", args.zh_config, "train", str(cache_dir))
    train_path = processed_dir / "train.txt"
    val_path = processed_dir / "val.txt"
    # 独立的随机数对象使文档级 train/val 划分可重复。
    rng = random.Random(args.seed)
    # max_chars 是原始清洗文本的字符预算，不保证最终 token 比例。
    # 实际训练中发现中文每字符需要更多 token，后来才改为按语言分层采样。
    language_paths = [(english, args.english_chars, "en"), (chinese, args.chinese_chars, "zh")]
    train_count = 0
    val_count = 0
    # 两个文件会被覆盖；每篇文档只写到其中一个文件，避免直接重复。
    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for iterator, max_chars, language in language_paths:
            used_chars = 0
            for document in iterator:
                # 清洗所有空白并排除很短的文档；原来的段落结构会丢失。
                text = " ".join(str(document).split())
                if len(text) < 20:
                    continue
                # 到达当前语言的预算后停止读取该语言，不继续找更短的文档。
                if used_chars + len(text) > max_chars:
                    break
                used_chars += len(text) + 1
                # 以 2% 概率把整篇文档送入验证集；不是逐 token 划分。
                if rng.random() < 0.02:
                    val_handle.write(text + "\n")
                    val_count += 1
                else:
                    train_handle.write(text + "\n")
                    train_count += 1
            print(f"Collected {used_chars:,} characters for {language}")
    print(f"Wrote {train_count} train documents and {val_count} validation documents")
    return train_path, val_path


def encode_file(text_path: Path, bin_path: Path, tokenizer_model: str) -> int:
    """逐行编码文本，把 ID 连续写为 uint16，返回 token 总数。"""
    tokenizer = load_tokenizer(tokenizer_model)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # "rb/wb" 中的 b 表示二进制；这里用 wb，每次运行会覆盖旧 .bin。
    with text_path.open("r", encoding="utf-8") as input_handle, bin_path.open("wb") as output_handle:
        for line in input_handle:
            # 每行代表一篇清洗后的文档；encode_text() 会在末尾追加 EOS。
            ids = encode_text(tokenizer, line.strip())
            # 词表只有 16000 个 ID，uint16（0 到 65535）足够保存。
            # tofile 直接追加原始整数，不保存文本和文档长度元数据。
            np.asarray(ids, dtype=np.uint16).tofile(output_handle)
            count += len(ids)
    print(f"Encoded {count:,} tokens to {bin_path}")
    return count


def main():
    """解析命令行选项，并按顺序执行语料准备的三个阶段。"""
    parser = argparse.ArgumentParser()
    # 路径默认相对于执行命令时的工作目录；PowerShell 包装脚本会先切到项目根目录。
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--tokenizer-model", default="tokenizer/spm.model")
    parser.add_argument("--zh-config", default="20231101.zh")
    parser.add_argument("--english-chars", type=int, default=180_000_000)
    parser.add_argument("--chinese-chars", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    # 重用已有文本可省去网络下载，但下方仍会重训 tokenizer 和重编码 .bin。
    # 重训 tokenizer 后 token ID 可能变化，不能随意配合旧模型 checkpoint。
    if args.skip_download:
        train_path = Path(args.processed_dir) / "train.txt"
        val_path = Path(args.processed_dir) / "val.txt"
    else:
        train_path, val_path = build_text_splits(args)

    # 只在训练文本上学习词表，避免验证集的信息进入 tokenizer 训练。
    train_sentencepiece([str(train_path)], args.tokenizer_model)
    encode_file(train_path, Path(args.processed_dir) / "train.bin", args.tokenizer_model)
    encode_file(val_path, Path(args.processed_dir) / "val.bin", args.tokenizer_model)


if __name__ == "__main__":
    # 直接以 python -m src.prepare_data 运行时进入 main；被 import 时不执行。
    main()
