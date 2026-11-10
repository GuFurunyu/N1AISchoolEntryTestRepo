"""Measure a short GPU training step before committing to a long run."""

import argparse
import time

import torch

from .config import load_config
from .model import TransformerLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    model_config, _, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    inputs = torch.randint(0, model_config.vocab_size, (train_config.micro_batch_size, model_config.block_size), device=device)
    targets = torch.randint(0, model_config.vocab_size, inputs.shape, device=device)
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = args.steps * inputs.numel()
    print(f"device={device}")
    print(f"steps={args.steps}")
    print(f"tokens={tokens:,}")
    print(f"tokens_per_second={tokens / max(elapsed, 1e-6):.2f}")
    if device.type == "cuda":
        print(f"peak_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.3f}")


if __name__ == "__main__":
    main()
