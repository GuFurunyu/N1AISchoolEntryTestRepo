"""SentencePiece training and encoding helpers."""

from pathlib import Path
from typing import Iterable, List


def train_sentencepiece(input_files: Iterable[str], model_prefix: str, vocab_size: int = 16_000):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install sentencepiece before training the tokenizer") from exc

    model_path = Path(model_prefix)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # SentencePiece appends .model and .vocab to model_prefix. Accept a
    # user-facing path such as tokenizer/spm.model without producing a
    # confusing tokenizer/spm.model.model filename.
    prefix = str(model_path.with_suffix("")) if model_path.suffix == ".model" else str(model_path)
    spm.SentencePieceTrainer.train(
        input=",".join(str(Path(path)) for path in input_files),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        normalization_rule_name="nmt_nfkc",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )


def load_tokenizer(model_path: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install sentencepiece before loading the tokenizer") from exc
    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(model_path)):
        raise RuntimeError(f"Could not load tokenizer: {model_path}")
    return tokenizer


def encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, out_type=int) + [tokenizer.eos_id()]
