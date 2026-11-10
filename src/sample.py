"""Generate text from a saved checkpoint."""

import argparse

import torch

from .config import load_config
from .model import TransformerLM
from .tokenizer import load_tokenizer


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature, top_k):
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    model_config, data_config, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = load_tokenizer(data_config.tokenizer_model)
    model = TransformerLM(model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    encoded = tokenizer.encode(args.prompt, out_type=int)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    output = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
