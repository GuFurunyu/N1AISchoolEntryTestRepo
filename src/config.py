"""集中定义模型、数据和训练参数，并从 YAML 文件读取用户配置。

这里的 dataclass 只是保存参数的容器，不会自己创建模型或启动训练。
实际运行时，train.py 先调用 load_config()，再把三组参数分别交给模型、
数据加载器和训练循环。
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """决定模型结构的参数；改变它们通常会使旧 checkpoint 无法直接加载。"""

    # Tokenizer 共有 16000 个 ID，因此 embedding 和输出 logits 的词表维都是 16000。
    vocab_size: int = 16_000
    # 一次前向传播允许的最大 token 数；注意不是字符数。
    block_size: int = 512
    # 串联的 Transformer block 数量。
    n_layer: int = 8
    # 每层 attention 的并行 head 数；每个 head 的维度是 n_embd / n_head。
    n_head: int = 8
    # 每个 token 在模型内部的向量维度，也叫 d_model。
    n_embd: int = 512
    # 训练期间随机置零部分激活的概率；验证和生成时关闭。
    dropout: float = 0.1
    # RoPE 位置旋转所用的频率基数。
    rope_base: float = 10_000.0
    # 是否给 QKV、attention 输出和 MLP 线性层添加可训练偏置。
    bias: bool = True


@dataclass
class DataConfig:
    """数据文件路径及中英文在连续 token 文件中的边界。"""

    # 文本与 token ID 之间的映射必须和训练 checkpoint 保持一致。
    tokenizer_model: str = "tokenizer/spm.model"
    # 两个 .bin 文件以 uint16 存储连续 token ID，读取时采用 memory map。
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"
    # 文件按英文在前、中文在后写入；这两个数字是实测英文 token 数。
    # 若重新准备语料或重训 tokenizer，需要重新统计并更新边界。
    # 双语旧方案使用语言边界；单语方案将两个值设为 0，表示整个文件都是英文。
    train_english_tokens: int = 0
    val_english_tokens: int = 0
    # 仅在双语旧方案中使用；单语配置不启用分层采样。
    english_sample_probability: float = 0.74


@dataclass
class TrainConfig:
    """训练过程、日志频率和 checkpoint 保存频率的参数。"""

    # 固定 Python、NumPy 和 PyTorch 随机数的起点，便于重复实验。
    seed: int = 1337
    # 请求使用 CUDA；不可用时 train.py 会退回 CPU。
    device: str = "cuda"
    # 记录预期精度；当前 train.py 的 autocast 实际固定使用 float16。
    amp_dtype: str = "float16"
    # 单次 forward/backward 的序列数，不等于一次优化器更新的总序列数。
    micro_batch_size: int = 4
    # 累积几次 micro-batch 的梯度后，才调用一次 AdamW 更新。
    grad_accum_steps: int = 4
    # 训练目标是“看过的 token 次数”，语料可被重复抽样。
    max_tokens: int = 80_000_000
    # 可选的优化器步数上限；None 表示只用 max_tokens 推算终点。
    max_steps: Optional[int] = None
    # warmup 结束后的最高学习率和余弦下降的最低学习率。
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    # 学习率从低值线性升到最高值所用的优化器步数。
    warmup_steps: int = 500
    # AdamW 的权重衰减；当前实现对所有参数统一使用该值。
    weight_decay: float = 0.1
    # AdamW 一阶矩和二阶矩的指数移动平均系数。
    beta1: float = 0.9
    beta2: float = 0.95
    # 每次更新前允许的最大梯度范数；超过时整体按比例缩小。
    grad_clip: float = 1.0
    # 以下 interval 均以训练循环的 step 计，而不是 micro-batch 次数。
    # 混合精度溢出时优化器可能跳过更新，但循环 step 仍增加。
    log_interval: int = 20
    eval_interval: int = 250
    checkpoint_interval: int = 1_000
    # 每种语言验证时读取多少个 batch。
    eval_batches: int = 50
    # Windows 上默认用主进程加载数据，避免多进程复制/启动开销。
    num_workers: int = 0
    # 权重与训练状态、TensorBoard 与 JSONL 日志的输出目录。
    output_dir: str = "checkpoints"
    log_dir: str = "runs"


def _merge_dataclass(cls, values: Optional[Dict[str, Any]]):
    """用 YAML 中认识的字段覆盖 dataclass 默认值，返回配置对象。

    未提供的字段沿用 dataclass 默认值。当前实现会忽略拼错或未知的字段，
    因此修改 YAML 后应核对程序启动时打印出的关键参数。
    """
    values = values or {}
    # __dataclass_fields__ 列出该配置类声明过的字段名。
    allowed = {field for field in cls.__dataclass_fields__}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str):
    """读取 YAML，依次返回 ModelConfig、DataConfig、TrainConfig。"""
    try:
        import yaml
    except ImportError as exc:
        # 保留原始异常作为原因，使用户知道需要安装哪个包。
        raise RuntimeError("PyYAML is required to load a training config") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        # 空 YAML 文件会解析为 None；这里转为空字典以使用全部默认值。
        raw = yaml.safe_load(handle) or {}
    return (
        _merge_dataclass(ModelConfig, raw.get("model")),
        _merge_dataclass(DataConfig, raw.get("data")),
        _merge_dataclass(TrainConfig, raw.get("train")),
    )


def config_as_dict(model: ModelConfig, data: DataConfig, train: TrainConfig):
    """把三个配置对象转为普通字典，便于写入 checkpoint。"""
    return {"model": asdict(model), "data": asdict(data), "train": asdict(train)}
