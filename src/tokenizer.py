"""训练、加载 SentencePiece，并把一篇文本转换为 token ID 序列。

SentencePiece 负责决定文本如何切分；Transformer 本身只接收整数 ID，
不知道原始输入中的汉字、单词或标点是什么。
"""

from pathlib import Path
from typing import Iterable, List


def train_sentencepiece(input_files: Iterable[str], model_prefix: str, vocab_size: int = 16_000):
    """仅从训练文本学习切分规则，生成 .model 和 .vocab 两个文件。

input_files 可以包含多个训练文本路径；model_prefix 是输出文件的前缀。
例如前缀 tokenizer/spm 会得到 tokenizer/spm.model 和 tokenizer/spm.vocab。
"""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install sentencepiece before training the tokenizer") from exc

    model_path = Path(model_prefix)
    # 确保 tokenizer/ 这样的父目录存在；exist_ok=True 允许重复执行。
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # SentencePiece 会自行追加 .model 和 .vocab。如果调用者已经写了 .model，
    # 先去掉后缀，否则真实事故中曾出现 spm.model.model，后续加载找不到文件。
    prefix = str(model_path.with_suffix("")) if model_path.suffix == ".model" else str(model_path)
    spm.SentencePieceTrainer.train(
        # 库接受逗号连接的多个输入文件路径。
        input=",".join(str(Path(path)) for path in input_files),
        model_prefix=prefix,
        vocab_size=vocab_size,
        # Unigram 模型从候选片段中选择较可能的切分；这里没有使用 BPE。
        model_type="unigram",
        # 保留训练文本中累计覆盖 99.95% 的字符；极少见字符可能变成 <unk>。
        character_coverage=0.9995,
        # 统一部分字符形式，例如全角与半角的兼容形式。
        normalization_rule_name="nmt_nfkc",
        # 四个特殊 token 的 ID 固定下来；当前训练数据只主动追加 EOS。
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        # 最多抽取这么多句子训练 tokenizer，避免 tokenizer 训练无限膨胀。
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        # 语料不足时允许实际词表略小于要求；当前产物实测为 16000。
        hard_vocab_limit=False,
    )


def load_tokenizer(model_path: str):
    """从 .model 文件恢复训练时的切分规则和 token-ID 映射。"""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install sentencepiece before loading the tokenizer") from exc
    tokenizer = spm.SentencePieceProcessor()
    # 模型权重必须搭配同一个 tokenizer；重训 tokenizer 后 ID 含义可能改变。
    if not tokenizer.load(str(model_path)):
        raise RuntimeError(f"Could not load tokenizer: {model_path}")
    return tokenizer


def encode_text(tokenizer, text: str) -> List[int]:
    """把单篇文本变成整数 ID，并用 EOS 标记文档结束。

    out_type=int 返回 Python 整数列表；EOS 让连续拼接的语料仍有边界信号。
    这里不在开头添加 BOS，后续训练窗口也不要求从文档开头开始。
    """
    return tokenizer.encode(text, out_type=int) + [tokenizer.eos_id()]
