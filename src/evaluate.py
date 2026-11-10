"""Evaluate validation loss and perplexity for a saved checkpoint."""

import argparse
import math

import torch

from .config import load_config
from .dataset import make_eval_loader
from .model import TransformerLM


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    model_config, data_config, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    losses = {}
    spans = {
        "english": (0, data_config.val_english_tokens - model_config.block_size),
        "chinese": (data_config.val_english_tokens, None),
    }
    for language, (start, end) in spans.items():
        loader = make_eval_loader(data_config.val_bin, model_config.block_size, train_config.micro_batch_size, args.batches, train_config.num_workers, start, end)
        iterator = iter(loader)
        total = 0.0
        for _ in range(args.batches):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(inputs, targets)
            total += float(loss)
        losses[language] = total / args.batches
        print(f"{language}_loss={losses[language]:.5f}")
    weighted = 0.65 * losses["english"] + 0.35 * losses["chinese"]
    print(f"validation_loss_65en_35zh={weighted:.5f}")
    print(f"perplexity_65en_35zh={math.exp(min(20.0, weighted)):.5f}")


if __name__ == "__main__":
    main()
