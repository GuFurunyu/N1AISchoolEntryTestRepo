"""Download, sample, split, tokenize, and encode the training corpus."""

import argparse
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .tokenizer import encode_text, train_sentencepiece, load_tokenizer


def _write_documents(documents: Iterable[str], output: Path, max_chars: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for document in documents:
            text = " ".join(str(document).split())
            if len(text) < 20:
                continue
            if written + len(text) > max_chars:
                break
            handle.write(text + "\n")
            written += len(text) + 1
            count += 1
    return count


def _load_documents(dataset_name: str, config: str, split: str, cache_dir: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before downloading the corpus") from exc
    if config:
        dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    text_key = "text" if "text" in dataset.column_names else "content"
    for row in dataset:
        value = row.get(text_key, "")
        if value:
            yield value


def build_text_splits(args) -> Tuple[Path, Path]:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    cache_dir = Path(args.cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    english = _load_documents("roneneldan/TinyStories", None, "train", str(cache_dir))
    chinese = _load_documents("wikimedia/wikipedia", args.zh_config, "train", str(cache_dir))
    train_path = processed_dir / "train.txt"
    val_path = processed_dir / "val.txt"
    rng = random.Random(args.seed)
    language_paths = [(english, args.english_chars, "en"), (chinese, args.chinese_chars, "zh")]
    train_count = 0
    val_count = 0
    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for iterator, max_chars, language in language_paths:
            used_chars = 0
            for document in iterator:
                text = " ".join(str(document).split())
                if len(text) < 20:
                    continue
                if used_chars + len(text) > max_chars:
                    break
                used_chars += len(text) + 1
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
    tokenizer = load_tokenizer(tokenizer_model)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with text_path.open("r", encoding="utf-8") as input_handle, bin_path.open("wb") as output_handle:
        for line in input_handle:
            ids = encode_text(tokenizer, line.strip())
            np.asarray(ids, dtype=np.uint16).tofile(output_handle)
            count += len(ids)
    print(f"Encoded {count:,} tokens to {bin_path}")
    return count


def main():
    parser = argparse.ArgumentParser()
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

    if args.skip_download:
        train_path = Path(args.processed_dir) / "train.txt"
        val_path = Path(args.processed_dir) / "val.txt"
    else:
        train_path, val_path = build_text_splits(args)

    train_sentencepiece([str(train_path)], args.tokenizer_model)
    encode_file(train_path, Path(args.processed_dir) / "train.bin", args.tokenizer_model)
    encode_file(val_path, Path(args.processed_dir) / "val.bin", args.tokenizer_model)


if __name__ == "__main__":
    main()
