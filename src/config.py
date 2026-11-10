"""Configuration objects shared by data preparation and training scripts."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    rope_base: float = 10_000.0
    bias: bool = True


@dataclass
class DataConfig:
    tokenizer_model: str = "tokenizer/spm.model"
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"
    train_english_tokens: int = 42_420_185
    val_english_tokens: int = 849_432
    english_sample_probability: float = 0.74


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda"
    amp_dtype: str = "float16"
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    max_tokens: int = 80_000_000
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_interval: int = 20
    eval_interval: int = 250
    checkpoint_interval: int = 1_000
    eval_batches: int = 50
    num_workers: int = 0
    output_dir: str = "checkpoints"
    log_dir: str = "runs"


def _merge_dataclass(cls, values: Optional[Dict[str, Any]]):
    values = values or {}
    allowed = {field for field in cls.__dataclass_fields__}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str):
    """Load a YAML file without making the rest of the code depend on YAML details."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load a training config") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return (
        _merge_dataclass(ModelConfig, raw.get("model")),
        _merge_dataclass(DataConfig, raw.get("data")),
        _merge_dataclass(TrainConfig, raw.get("train")),
    )


def config_as_dict(model: ModelConfig, data: DataConfig, train: TrainConfig):
    return {"model": asdict(model), "data": asdict(data), "train": asdict(train)}
