"""Training loop with mixed precision, evaluation, logging, checkpoints, and resume."""

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("CUDA requested but unavailable; using CPU")
    return torch.device("cpu")


def learning_rate(step: int, total_steps: int, train_config) -> float:
    if step < train_config.warmup_steps:
        return train_config.learning_rate * (step + 1) / max(1, train_config.warmup_steps)
    progress = min(1.0, (step - train_config.warmup_steps) / max(1, total_steps - train_config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_config.min_learning_rate + cosine * (train_config.learning_rate - train_config.min_learning_rate)


def latest_checkpoint(directory: Path) -> Optional[Path]:
    candidates = list(directory.glob("*.pt"))
    candidates = [path for path in candidates if re.search(r"step_(\d+)\.pt$", path.name)]
    candidates.sort(key=lambda path: int(re.search(r"step_(\d+)\.pt$", path.name).group(1)))
    return candidates[-1] if candidates else None


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, tokens_seen: int, config_dict: Dict):
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
    model.eval()
    total = 0.0
    count = 0
    iterator = iter(loader)
    for _ in range(batches):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, loss = model(inputs, targets)
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    model_config, data_config, train_config = load_config(args.config)
    if args.max_tokens is not None:
        train_config.max_tokens = args.max_tokens
    set_seed(train_config.seed)
    device = choose_device(train_config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = TransformerLM(model_config).to(device)
    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    effective_batch_tokens = train_config.micro_batch_size * train_config.grad_accum_steps * model_config.block_size
    total_steps = math.ceil(train_config.max_tokens / effective_batch_tokens)
    if train_config.max_steps is not None:
        total_steps = min(total_steps, train_config.max_steps)
    print(f"Effective tokens per optimizer step: {effective_batch_tokens:,}")
    print(f"Planned optimizer steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = Path(train_config.output_dir)
    log_dir = Path(train_config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        print("TensorBoard is unavailable; JSONL logging remains enabled")

    step = 0
    tokens_seen = 0
    checkpoint = latest_checkpoint(output_dir) if args.resume == "auto" else None
    if checkpoint is not None:
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
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"Resumed from {checkpoint} at step {step}")

    run_start_tokens = tokens_seen

    train_loader = make_loader(
        data_config.train_bin, model_config.block_size, train_config.micro_batch_size,
        True, train_config.num_workers, data_config.train_english_tokens,
        data_config.english_sample_probability,
    )
    en_val_loader = make_eval_loader(
        data_config.val_bin, model_config.block_size, train_config.micro_batch_size,
        train_config.eval_batches, train_config.num_workers,
        0, data_config.val_english_tokens - model_config.block_size,
    )
    zh_val_loader = make_eval_loader(
        data_config.val_bin, model_config.block_size, train_config.micro_batch_size,
        train_config.eval_batches, train_config.num_workers,
        data_config.val_english_tokens, None,
    )
    iterator = iter(train_loader)
    model.train()
    started = time.time()

    def record(payload):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        if writer is not None:
            for key, value in payload.items():
                if isinstance(value, (int, float)) and key not in {"step"}:
                    writer.add_scalar(key, value, payload["step"])

    try:
        while step < total_steps:
            current_lr = learning_rate(step, total_steps, train_config)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            # Four micro-batches contribute to one optimizer update.
            for _ in range(train_config.grad_accum_steps):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    inputs, targets = next(iterator)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    _, loss = model(inputs, targets)
                    # Averaging here keeps the accumulated gradient at the batch mean scale.
                    scaled_loss = loss / train_config.grad_accum_steps
                scaler.scale(scaled_loss).backward()
                loss_sum += float(loss.detach().item())
                tokens_seen += inputs.numel()

            # Restore gradient scale before clipping, then let AdamW update parameters.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % train_config.log_interval == 0 or step == 1:
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
                en_loss = evaluate(model, en_val_loader, device, train_config.eval_batches, amp_enabled)
                zh_loss = evaluate(model, zh_val_loader, device, train_config.eval_batches, amp_enabled)
                val_loss = 0.65 * en_loss + 0.35 * zh_loss
                payload = {
                    "step": step, "tokens_seen": tokens_seen,
                    "val_loss": val_loss, "val_loss_en": en_loss, "val_loss_zh": zh_loss,
                    "perplexity": math.exp(min(20.0, val_loss)),
                    "val_sampling": "stratified_65en_35zh_v3",
                }
                print(json.dumps(payload, ensure_ascii=True))
                record(payload)

            if step % train_config.checkpoint_interval == 0:
                save_checkpoint(output_dir / f"step_{step:08d}.pt", model, optimizer, scaler, step, tokens_seen, config_as_dict(model_config, data_config, train_config))

    except KeyboardInterrupt:
        print("Interrupted; saving a resumable checkpoint")
        save_checkpoint(output_dir / f"interrupt_step_{step:08d}.pt", model, optimizer, scaler, step, tokens_seen, config_as_dict(model_config, data_config, train_config))
        raise
    finally:
        if writer is not None:
            writer.close()

    save_checkpoint(output_dir / f"final_step_{step:08d}.pt", model, optimizer, scaler, step, tokens_seen, config_as_dict(model_config, data_config, train_config))


if __name__ == "__main__":
    main()
