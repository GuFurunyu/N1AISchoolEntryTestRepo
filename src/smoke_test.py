"""Run cheap correctness checks before downloading a large corpus."""

import argparse

import torch

from .config import load_config, ModelConfig
from .model import TransformerLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    model_config, _, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    print(f"device={device}")
    print(f"parameters={model.num_parameters():,}")

    batch, seq_len = 2, min(64, model_config.block_size)
    inputs = torch.randint(0, model_config.vocab_size, (batch, seq_len), device=device)
    targets = torch.randint(0, model_config.vocab_size, (batch, seq_len), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    initial_loss = None
    final_loss = None
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets)
        if initial_loss is None:
            initial_loss = float(loss)
        loss.backward()
        optimizer.step()
        final_loss = float(loss)
    assert logits.shape == (batch, seq_len, model_config.vocab_size)
    assert torch.isfinite(loss)
    print(f"initial_loss={initial_loss:.5f}")
    print(f"final_loss={final_loss:.5f}")
    model.eval()
    changed_future = inputs.clone()
    changed_future[:, seq_len // 2 :] = torch.randint(
        0, model_config.vocab_size, changed_future[:, seq_len // 2 :].shape, device=device
    )
    with torch.no_grad():
        original_logits, _ = model(inputs)
        changed_logits, _ = model(changed_future)
    assert torch.allclose(
        original_logits[:, : seq_len // 2], changed_logits[:, : seq_len // 2],
        atol=1e-5, rtol=1e-5,
    ), "Future tokens changed earlier logits; check the causal mask"
    print("causal_mask_check=PASS")
    print("forward_backward_check=PASS")


if __name__ == "__main__":
    main()
