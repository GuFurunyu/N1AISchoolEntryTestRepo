"""KV cache 对照实验：同一模型、同一输入，只改变推理时是否复用 K/V。

运行：python -m src.inference_experiment --checkpoint checkpoints/final_step_00029297.pt
结果写入 JSON，便于以后重新计算速度比或检查输出。基准提示词通过
重复并截断固定文本构造，重点考察计算成本；生成内容不是质量评分。
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from .config import load_config
from .model import TransformerLM
from .sample import generate
from .tokenizer import load_tokenizer


PROMPTS = {
    "english": "Once upon a time, a curious child found a small wooden door in the garden. ",
    "chinese": "从前，一个好奇的孩子在花园里发现了一扇小木门。他轻轻推开门，看到了一条陌生的小路。",
}


def file_sha256(path: Path) -> str:
    """逐块读取检查点计算哈希值，避免一次把整个文件装入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_prompt_ids(tokenizer, text: str, target_tokens: int, device: torch.device) -> torch.Tensor:
    """重复固定句子后截到精确 token 数，让两个长度条件可比较。"""
    ids = tokenizer.encode(text * (target_tokens + 1), out_type=int)[:target_tokens]
    if len(ids) != target_tokens:
        raise ValueError("Prompt text did not produce enough tokens")
    return torch.tensor([ids], dtype=torch.long, device=device)


def synchronize(device: torch.device):
    """GPU 异步执行；计时边界必须等它完成。CPU 不需要同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def compare_logits(model, input_ids: torch.Tensor) -> dict:
    """比较同一上下文的下一 token logits，再比较追加一个 token 后的结果。

第二次调用缓存路径时只送入新增 token，可直接检验位置偏移和 K/V 拼接。
这里用绝对差及预测 ID 衡量数值一致性，不把微小浮点误差当成质量变化。
"""
    full_logits, _ = model(input_ids)
    cached_logits, cache = model.forward_cached(input_ids)
    first_diff = (full_logits[:, -1] - cached_logits[:, -1]).abs().max().item()
    next_id = full_logits[:, -1].argmax(dim=-1, keepdim=True)
    extended = torch.cat((input_ids, next_id), dim=1)
    full_next, _ = model(extended)
    cached_next, _ = model.forward_cached(next_id, cache)
    second_diff = (full_next[:, -1] - cached_next[:, -1]).abs().max().item()
    return {
        "prefill_max_abs_logit_diff": first_diff,
        "incremental_max_abs_logit_diff": second_diff,
        "next_token_equal": bool(
            torch.equal(full_next[:, -1].argmax(-1), cached_next[:, -1].argmax(-1))
        ),
    }


@torch.inference_mode()
def timed_generation(model, input_ids: torch.Tensor, new_tokens: int, use_cache: bool) -> dict:
    """记录一次完整生成的耗时、速度和 PyTorch 峰值显存。

包含首轮提示词处理和逐 token 解码，也包含相同的 top-k=1 采样开销。
top-k=1 使选词确定，输出不同就应检查缓存实现或数值边界。
"""
    synchronize(input_ids.device)
    before_bytes = torch.cuda.memory_allocated() if input_ids.is_cuda else None
    if input_ids.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = generate(model, input_ids, new_tokens, temperature=1.0, top_k=1, use_kv_cache=use_cache)
    synchronize(input_ids.device)
    elapsed = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated() if input_ids.is_cuda else None
    result = {
        "seconds": elapsed,
        "generated_tokens_per_second": new_tokens / elapsed,
        "peak_allocated_mib": peak_bytes / 1024**2 if peak_bytes is not None else None,
        "peak_extra_mib": (peak_bytes - before_bytes) / 1024**2 if peak_bytes is not None else None,
        "new_token_ids": output[0, input_ids.size(1) :].tolist(),
    }
    return result


def main():
    """固定实验条件，交替测量两种路径，并保存机器可读记录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/final_step_00029297.pt")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", default="notes/kv-cache-results.json")
    args = parser.parse_args()
    if args.new_tokens < 2 or args.trials < 1:
        parser.error("--new-tokens must be >= 2 and --trials must be >= 1")

    model_config, data_config, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(data_config.tokenizer_model)
    model = TransformerLM(model_config).to(device)
    # 训练检查点还含 AdamW 状态；只让模型权重进入 GPU。
    # 初版曾把整个文件映射到 GPU，导致推理峰值显存虚高。
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    checkpoint_step = state.get("step")
    del state
    model.eval()

    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": checkpoint_step,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "block_size": model_config.block_size,
        "new_tokens": args.new_tokens,
        "trials": args.trials,
        "decoding": "top_k=1, temperature=1.0, batch=1, torch.inference_mode",
        "cases": [],
    }

    # 两种语言、两种提示词长度。确保最长条件仍能至少增量解码一个 token。
    for language, text in PROMPTS.items():
        for prompt_tokens in (32, 256):
            if prompt_tokens + args.new_tokens > model_config.block_size:
                parser.error("Prompt plus generated tokens exceeds block_size; reduce --new-tokens")
            input_ids = make_prompt_ids(tokenizer, text, prompt_tokens, device)
            case = {
                "language": language,
                "prompt_tokens": prompt_tokens,
                "prompt_text": tokenizer.decode(input_ids[0].tolist()),
                "logits": compare_logits(model, input_ids),
                "uncached": [],
                "cached": [],
            }
            # 预热必须使用和正式测量相同的生成长度。初版只预热 8 token，
            # 首次 128-token 运行有明显额外开销，使短提示词的结果波动。
            for mode in (False, True):
                timed_generation(model, input_ids, args.new_tokens, mode)
            # 每轮交换先后顺序，减少运行顺序带来的系统性偏差。
            for trial in range(args.trials):
                for mode in ((False, True) if trial % 2 == 0 else (True, False)):
                    key = "cached" if mode else "uncached"
                    case[key].append(timed_generation(model, input_ids, args.new_tokens, mode))
            uncached_ids = case["uncached"][0]["new_token_ids"]
            case["outputs_equal"] = all(
                run["new_token_ids"] == uncached_ids
                for key in ("uncached", "cached")
                for run in case[key]
            )
            case["generated_text"] = tokenizer.decode(uncached_ids)
            case["median_uncached_seconds"] = statistics.median(
                run["seconds"] for run in case["uncached"]
            )
            case["median_cached_seconds"] = statistics.median(
                run["seconds"] for run in case["cached"]
            )
            case["speedup"] = case["median_uncached_seconds"] / case["median_cached_seconds"]
            print(
                f"{language} prompt={prompt_tokens} output_equal={case['outputs_equal']} "
                f"uncached={case['median_uncached_seconds']:.3f}s "
                f"cached={case['median_cached_seconds']:.3f}s speedup={case['speedup']:.2f}x"
            )
            record["cases"].append(case)

    # 跨过最大上下文后，缓存必须从最近 512 个 token 重建。
    # 这项只检查输出一致性，不纳入上述性能对照（重建会抵消加速）。
    overflow_input = make_prompt_ids(
        tokenizer, PROMPTS["english"], model_config.block_size - 4, device
    )
    overflow_uncached = generate(
        model, overflow_input, 8, temperature=1.0, top_k=1, use_kv_cache=False
    )
    overflow_cached = generate(
        model, overflow_input, 8, temperature=1.0, top_k=1, use_kv_cache=True
    )
    record["context_overflow_check"] = {
        "prompt_tokens": overflow_input.size(1),
        "new_tokens": 8,
        "outputs_equal": bool(torch.equal(overflow_uncached, overflow_cached)),
    }
    print(f"context_overflow_output_equal={record['context_overflow_check']['outputs_equal']}")

    # 额外检查真实的随机采样设置。两种路径分别重置同一 RNG 种子；
    # 若输出不同，需要检查数值差异是否跨过抽样概率边界。
    record["stochastic_checks"] = []
    for language, text in PROMPTS.items():
        ids = torch.tensor([tokenizer.encode(text, out_type=int)], dtype=torch.long, device=device)
        outputs = []
        with torch.inference_mode():
            for mode in (False, True):
                torch.manual_seed(1337)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(1337)
                outputs.append(generate(model, ids, 64, temperature=0.8, top_k=50, use_kv_cache=mode))
        check = {
            "language": language,
            "prompt": text,
            "prompt_tokens": ids.size(1),
            "seed": 1337,
            "temperature": 0.8,
            "top_k": 50,
            "new_tokens": 64,
            "outputs_equal": bool(torch.equal(outputs[0], outputs[1])),
            "generated_text": tokenizer.decode(outputs[0][0, ids.size(1) :].tolist()),
        }
        print(f"stochastic_{language}_output_equal={check['outputs_equal']}")
        record["stochastic_checks"].append(check)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved experiment record: {output}")


if __name__ == "__main__":
    main()
