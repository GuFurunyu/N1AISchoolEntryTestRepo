"""训练入口：串起前向计算、反向传播、参数更新、验证、日志和断点续训。

运行方式：在项目根目录执行 python -m src.train --config configs/base.yaml。
模型结构在 model.py 中；本文件负责组织训练过程。
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import config_as_dict, load_config
from .dataset import make_loader, make_eval_loader
from .model import TransformerLM


def set_seed(seed: int):
    """设置 Python、NumPy、PyTorch 的随机种子，便于复现。

GPU 底层算法及数据加载等因素仍可能导致重复运行不完全相同。
"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """配置要求 CUDA 且当前环境可用时使用 GPU，否则使用 CPU。"""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("CUDA requested but unavailable; using CPU")
    return torch.device("cpu")


def learning_rate(step: int, total_steps: int, train_config) -> float:
    """计算当前学习率：先线性预热，再沿余弦曲线降到最小学习率。

step 从 0 开始，代表即将执行的训练循环索引；total_steps 是计划总步数。
"""
    if step < train_config.warmup_steps:
        return train_config.learning_rate * (step + 1) / max(1, train_config.warmup_steps)
    progress = min(1.0, (step - train_config.warmup_steps) / max(1, total_steps - train_config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_config.min_learning_rate + cosine * (train_config.learning_rate - train_config.min_learning_rate)


def latest_checkpoint(directory: Path) -> Optional[Path]:
    """按文件名中的 step 数字找最新检查点；找不到就返回 None。

普通、interrupt、final 文件都可能匹配；按步数而非修改时间选择。
"""
    candidates = list(directory.glob("*.pt"))
    candidates = [path for path in candidates if re.search(r"step_(\d+)\.pt$", path.name)]
    candidates.sort(key=lambda path: int(re.search(r"step_(\d+)\.pt$", path.name).group(1)))
    return candidates[-1] if candidates else None


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, tokens_seen: int, config_dict: Dict):
    """保存可续训状态：模型参数、AdamW 状态、混合精度状态及随机数状态。

数据加载器的当前位置没有保存，因此续训取到的数据批次不保证完全接续。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config_dict,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


@torch.no_grad()
def evaluate(model, loader, device: torch.device, batches: int, amp_enabled: bool):
    """估计指定数量验证批次的平均 loss，不构建反向传播计算图。"""
    model.eval()
    total = 0.0
    count = 0
    iterator = iter(loader)
    for _ in range(batches):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            # 验证迭代器走完时重新开始，保证凑满要求的批次数。
            iterator = iter(loader)
            inputs, targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, loss = model(inputs, targets)
        total += float(loss.item())
        count += 1
    model.train()
    # 恢复训练模式，令 dropout 等层恢复训练行为。
    return total / max(1, count)


def main():
    """解析参数，准备模型和数据，然后执行训练及周期性验证。"""
    parser = argparse.ArgumentParser()
    # --resume auto 搜索已有检查点；none 从头启动。
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    model_config, data_config, train_config = load_config(args.config)
    # 命令行可临时覆盖配置文件中的训练 token 预算。
    if args.max_tokens is not None:
        train_config.max_tokens = args.max_tokens
    set_seed(train_config.seed)
    device = choose_device(train_config.device)
    if device.type == "cuda":
        # GPU 上允许较快的矩阵乘法，并为固定尺寸输入挑选 cuDNN 实现。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = TransformerLM(model_config).to(device)
    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    # 一次参数更新处理的 token 数 = 小批样本数 * 梯度累积次数 * 序列长度。
    effective_batch_tokens = train_config.micro_batch_size * train_config.grad_accum_steps * model_config.block_size
    total_steps = math.ceil(train_config.max_tokens / effective_batch_tokens)
    if train_config.max_steps is not None:
        total_steps = min(total_steps, train_config.max_steps)
    print(f"Effective tokens per optimizer step: {effective_batch_tokens:,}")
    print(f"Planned optimizer steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(
        # AdamW 根据梯度的一阶/二阶移动平均更新参数，并施加权重衰减。
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    amp_enabled = device.type == "cuda"
    # 半精度可能让很小的梯度下溢；GradScaler 放大 loss 后再还原梯度。
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = Path(train_config.output_dir)
    log_dir = Path(train_config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    writer = None
    try:
        # TensorBoard 用于图形查看；缺少它时仍写 JSONL 日志。
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        print("TensorBoard is unavailable; JSONL logging remains enabled")

    step = 0
    tokens_seen = 0
    checkpoint = latest_checkpoint(output_dir) if args.resume == "auto" else None
    if checkpoint is not None:
        # 当前结构和数据配置仍来自 --config，应与检查点保持兼容。
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step = int(state.get("step", 0))
        tokens_seen = int(state.get("tokens_seen", 0))
        if state.get("rng_state") is not None:
            torch.set_rng_state(state["rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            # map_location=device 会把检查点中的 ByteTensor 映射到 GPU；
            # set_rng_state_all 要求接收 CPU ByteTensor，因此恢复前逐个移回 CPU。
            torch.cuda.set_rng_state_all(
                [item.cpu() for item in state["cuda_rng_state"]]
            )
        print(f"Resumed from {checkpoint} at step {step}")

    run_start_tokens = tokens_seen

    # 训练 loader 抽取训练窗口；单语配置使用整个文件，双语旧配置才按语言采样。
    train_loader = make_loader(
        data_config.train_bin, model_config.block_size, train_config.micro_batch_size,
        True, train_config.num_workers, data_config.train_english_tokens,
        data_config.english_sample_probability,
    )
    val_start = data_config.val_english_tokens if data_config.val_english_tokens > 0 else 0
    val_end = data_config.val_english_tokens - model_config.block_size if data_config.val_english_tokens > 0 else None
    val_loader = make_eval_loader(
        data_config.val_bin, model_config.block_size, train_config.micro_batch_size,
        train_config.eval_batches, train_config.num_workers,
        val_start, val_end,
    )
    iterator = iter(train_loader)
    model.train()
    started = time.time()

    def record(payload):
        """追加一行 JSON 日志，并把数值指标送入 TensorBoard。"""
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        if writer is not None:
            for key, value in payload.items():
                if isinstance(value, (int, float)) and key not in {"step"}:
                    writer.add_scalar(key, value, payload["step"])

    try:
        while step < total_steps:
            # 一轮循环计划执行一次参数更新。先设置学习率，再清除上一轮梯度。
            current_lr = learning_rate(step, total_steps, train_config)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            # 显存一次只需容纳一个小批；多次 backward 会把梯度加到同一组参数上。
            for _ in range(train_config.grad_accum_steps):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    # 数据迭代器耗尽后创建新迭代器，继续获取训练窗口。
                    iterator = iter(train_loader)
                    inputs, targets = next(iterator)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    # 前向传播得到各位置的词表分数，以及与 targets 对比的平均 loss。
                    _, loss = model(inputs, targets)
                    # 除以累积次数，令累积梯度维持在大批平均梯度的量级。
                    scaled_loss = loss / train_config.grad_accum_steps
                # backward 按链式法则求导，结果累加到每个模型参数的 .grad。
                scaler.scale(scaled_loss).backward()
                loss_sum += float(loss.detach().item())
                tokens_seen += inputs.numel()

            # 还原梯度数值，再限制全局梯度范数，避免单次更新过大。
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            # 检测到半精度溢出时 scaler 可能跳过 optimizer.step，但循环步数仍加一。
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % train_config.log_interval == 0 or step == 1:
                # 速度只统计本次进程启动后处理的 token，不含续训前的部分。
                elapsed = max(1e-6, time.time() - started)
                payload = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": loss_sum / train_config.grad_accum_steps,
                    "learning_rate": current_lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": (tokens_seen - run_start_tokens) / elapsed,
                }
                print(json.dumps(payload, ensure_ascii=True))
                record(payload)

            if step % train_config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, train_config.eval_batches, amp_enabled)
                payload = {
                    "step": step, "tokens_seen": tokens_seen,
                    "val_loss": val_loss, "val_loss_en": val_loss,
                    "val_sampling": "uniform_english_v1" if data_config.train_english_tokens == 0 else "stratified_65en_35zh_v3",
                    "perplexity": math.exp(min(20.0, val_loss)),
                }
                print(json.dumps(payload, ensure_ascii=True))
                record(payload)

            if step % train_config.checkpoint_interval == 0:
                # 定期保存完整训练状态，便于断点续训。
                save_checkpoint(output_dir / f"step_{step:08d}.pt", model, optimizer, scaler, step, tokens_seen, config_as_dict(model_config, data_config, train_config))

    except KeyboardInterrupt:
        # 收到 Ctrl+C 时尽量保存可恢复状态，再把中断交给调用方。
        print("Interrupted; saving a resumable checkpoint")
        save_checkpoint(output_dir / f"interrupt_step_{step:08d}.pt", model, optimizer, scaler, step, tokens_seen, config_as_dict(model_config, data_config, train_config))
        raise
    finally:
        # 正常结束或异常退出都要关闭日志写入器。
        if writer is not None:
            writer.close()

    # 达到计划步数后保存最终模型。
    save_checkpoint(output_dir / f"final_step_{step:08d}.pt", model, optimizer, scaler, step, tokens_seen, config_as_dict(model_config, data_config, train_config))


if __name__ == "__main__":
    # 作为模块直接运行才执行 main；被其他模块导入时不会自动开始训练。
    main()
