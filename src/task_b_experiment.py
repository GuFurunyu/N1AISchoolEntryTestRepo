"""任务 B 的可续跑训练器：两段训练、四条分叉路径、同目标验证。

运行入口：python -m src.task_b_experiment
本文件只使用项目原有模型与语料；实验设置在 configs/task_b/ 下。
训练权重写到已被 git 忽略的 checkpoints/task_b/，指标写到 experiments/task_b/。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .config import config_as_dict, load_config
from .model import TransformerLM
from .tokenizer import load_tokenizer
from .train import set_seed


STAGE_FILES = (
    "stage1/128.yaml",
    "stage1/512.yaml",
    "stage2/128-128.yaml",
    "stage2/128-512.yaml",
    "stage2/512-512.yaml",
    "stage2/512-128.yaml",
)


def read_yaml(path: Path) -> dict:
    """使用结构化 YAML 解析器读取配置，并拒绝空配置。"""
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def save_json(path: Path, value: dict):
    """先写临时文件，再替换结果文件，避免中断时留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path: Path, state: dict):
    """原子替换本阶段唯一的完整检查点，不持续堆积大文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """分块计算文件摘要，确认实验期间语料和 tokenizer 没变。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_config_digest(shared: dict, specs: list[dict], base: dict) -> str:
    """用排序后的完整设置生成固定摘要，防止错误地续跑不同实验。"""
    payload = json.dumps(
        {"shared": shared, "specs": specs, "base": base},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def learning_rate(completed_steps: int, total_steps: int, shared: dict) -> float:
    """所有路径共用的预热加余弦衰减日程；分叉后不重新开始计时。"""
    warmup = int(shared["warmup_steps"])
    peak = float(shared["peak_learning_rate"])
    minimum = float(shared["min_learning_rate"])
    if completed_steps < warmup:
        return peak * (completed_steps + 1) / max(1, warmup)
    progress = min(1.0, (completed_steps - warmup) / max(1, total_steps - warmup))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def read_windows(tokens: np.memmap, starts: np.ndarray, base_context: int) -> np.ndarray:
    """读取每个起点后连续的 513 个 token，最后一个只作为预测目标。"""
    return np.stack(
        [
            np.asarray(tokens[int(start) : int(start) + base_context + 1], dtype=np.int64)
            for start in starts
        ]
    )


def split_windows(windows: np.ndarray, context: int, base_context: int) -> tuple[np.ndarray, np.ndarray]:
    """相同原始窗口按 128 或 512 切段；输入与右移目标的扁平顺序相同。"""
    base_batch = windows.shape[0]
    pieces = base_context // context
    inputs = windows[:, :-1].reshape(base_batch * pieces, context)
    targets = windows[:, 1:].reshape(base_batch * pieces, context)
    return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)


@torch.inference_mode()
def evaluate(model, val_tokens: np.memmap, starts: np.ndarray, shared: dict, device: torch.device) -> dict:
    """固定中文窗口和最后 128 个目标，比较不同可见历史的交叉熵。"""
    model.eval()
    max_context = int(shared["model_max_context"])
    tail = int(shared["evaluation_target_tail"])
    batch_size = int(shared["base_windows_per_micro"])
    results = {}
    for visible_context in shared["evaluation_contexts"]:
        per_window = []
        for offset in range(0, len(starts), batch_size):
            windows = read_windows(val_tokens, starts[offset : offset + batch_size], max_context)
            input_ids = torch.tensor(
                np.ascontiguousarray(windows[:, max_context - visible_context : max_context]),
                dtype=torch.long,
                device=device,
            )
            target_ids = torch.tensor(
                np.ascontiguousarray(windows[:, max_context + 1 - tail : max_context + 1]),
                dtype=torch.long,
                device=device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(input_ids)
            # 只截取相同的最后 tail 个位置；长历史前面的目标不参与比较。
            losses = F.cross_entropy(
                logits[:, -tail:, :].float().reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="none",
            )
            per_window.extend(losses.reshape(-1, tail).mean(dim=1).cpu().tolist())
        results[str(visible_context)] = {
            "mean_nll": float(np.mean(per_window)),
            "per_window_nll": per_window,
            "target_tokens": len(per_window) * tail,
        }
    model.train()
    return results


def select_prompts(text_path: Path, tokenizer, shared: dict) -> list[dict]:
    """从验证文档选固定长 prompt；双语旧方案优先中文，单语方案使用英文。"""
    selected = []
    needed = int(shared["model_max_context"]) + 128
    lines = text_path.read_text(encoding="utf-8").splitlines()
    has_chinese = any(re.search("[\u3400-\u9fff]", line) for line in lines)
    for line_number, line in enumerate(lines, start=1):
        if has_chinese and re.search("[\u3400-\u9fff]", line) is None:
            continue
        ids = tokenizer.encode(line.strip(), out_type=int)
        if len(ids) < needed:
            continue
        prompt_ids = ids[: int(shared["model_max_context"])]
        selected.append({
            "validation_line": line_number,
            "prompt_ids": prompt_ids,
            "real_next_128_text": tokenizer.decode(ids[needed - 128 : needed]),
        })
        if len(selected) == int(shared["generation_prompts"]):
            return selected
    raise RuntimeError("Not enough long validation documents for generation prompts")


@torch.inference_mode()
def generate_samples(model, prompts: list[dict], tokenizer, shared: dict, device: torch.device) -> list[dict]:
    """用同一真实中文提示词的不同后缀自由生成；结果仅供定性审阅。"""
    model.eval()
    samples = []
    new_tokens = int(shared["generation_new_tokens"])
    for prompt_index, prompt in enumerate(prompts):
        full_prompt = torch.tensor([prompt["prompt_ids"]], dtype=torch.long, device=device)
        for visible_context in shared["evaluation_contexts"]:
            # 重置采样 RNG；不同条件下分布会变，固定种子不意味着输出应相同。
            seed = int(shared["generation_seed"]) + prompt_index
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            output = full_prompt
            for _ in range(new_tokens):
                context = output[:, -visible_context:]
                logits, _ = model(context)
                logits = logits[:, -1, :] / float(shared["generation_temperature"])
                values, indices = torch.topk(logits, int(shared["generation_top_k"]))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                probabilities = F.softmax(filtered, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
                output = torch.cat((output, next_id), dim=1)
            generated_ids = output[0, full_prompt.size(1) :].tolist()
            samples.append({
                "prompt_index": prompt_index,
                "validation_line": prompt["validation_line"],
                "visible_context": visible_context,
                "prompt_suffix_text": tokenizer.decode(prompt["prompt_ids"][-visible_context:]),
                "real_next_128_text": prompt["real_next_128_text"],
                "generated_text": tokenizer.decode(generated_ids),
                "generated_ids": generated_ids,
                "seed": seed,
            })
    return samples


def metric_event(stage: str, run_id: str, context: int, stage_step: int, global_step: int, metrics: dict) -> dict:
    """把三种验证损失放进同一条事件，便于 CSV 和 TensorBoard 同轴比较。"""
    nll128 = metrics["128"]["mean_nll"]
    nll256 = metrics["256"]["mean_nll"]
    nll512 = metrics["512"]["mean_nll"]
    return {
        "kind": "evaluation",
        "stage": stage,
        "run_id": run_id,
        "train_context": context,
        "stage_step": stage_step,
        "global_step": global_step,
        "tokens_seen": global_step * 8192,
        "nll_128": nll128,
        "nll_256": nll256,
        "nll_512": nll512,
        "long_context_gain": nll128 - nll512,
        "per_window_nll": {key: item["per_window_nll"] for key, item in metrics.items()},
    }


def write_history(path: Path, history: list[dict]):
    """从检查点保存的规范历史重建 JSONL，避免续跑造成重复行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for event in history:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def add_tensorboard(writer: SummaryWriter, event: dict):
    """每条最终路径都写入完整前后两阶段曲线，四条线可直接叠加。"""
    step = event["global_step"]
    if event["kind"] == "evaluation":
        for length in (128, 256, 512):
            writer.add_scalar(f"validation/nll_context_{length}", event[f"nll_{length}"], step)
        writer.add_scalar("validation/long_context_gain", event["long_context_gain"], step)
    else:
        writer.add_scalar("training/loss", event["train_loss"], step)
        writer.add_scalar("training/learning_rate", event["learning_rate"], step)
        writer.add_scalar("training/tokens_per_second", event["tokens_per_second"], step)
        writer.add_scalar("training/skipped_updates", event["skipped_updates"], step)


def update_comparison(results_root: Path):
    """将六个阶段的评估汇总成一张长表和一张宽表，Excel 可直接筛选/画图。"""
    rows = []
    for phase, run_id in (
        ("stage1", "parent128"), ("stage1", "parent512"),
        ("stage2", "128-128"), ("stage2", "128-512"),
        ("stage2", "512-512"), ("stage2", "512-128"),
    ):
        path = results_root / phase / run_id / "metrics.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event["kind"] != "evaluation":
                    continue
                rows.append({
                    "path": run_id,
                    "parent": run_id.split("-")[0] if phase == "stage2" else "",
                    "phase": event["stage"],
                    "global_step": event["global_step"],
                    "tokens_seen": event["tokens_seen"],
                    "nll_128": event["nll_128"],
                    "nll_256": event["nll_256"],
                    "nll_512": event["nll_512"],
                    "long_context_gain": event["long_context_gain"],
                })
    fields = ["path", "parent", "phase", "global_step", "tokens_seen",
              "nll_128", "nll_256", "nll_512", "long_context_gain"]
    long_path = results_root / "comparison.csv"
    with long_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["global_step"], row["path"])))
    # 一行代表同一个训练步，四条路径各有自己的列，便于 Excel 直接画折线。
    paths = ("128-128", "128-512", "512-512", "512-128")
    metrics = ("nll_128", "nll_256", "nll_512", "long_context_gain")
    by_step = {}
    for row in rows:
        if row["path"] not in paths:
            continue
        step = row["global_step"]
        by_step.setdefault(step, {"global_step": step, "tokens_seen": row["tokens_seen"]})
        for metric in metrics:
            by_step[step][f"{row['path']}__{metric}"] = row[metric]
    wide_fields = ["global_step", "tokens_seen"] + [f"{path}__{metric}" for path in paths for metric in metrics]
    with (results_root / "comparison_wide.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(by_step[step] for step in sorted(by_step))


def checkpoint_path(checkpoint_root: Path, spec: dict) -> Path:
    """按阶段和路径名隔离权重；绝不写入原模型的 checkpoints 根目录文件。"""
    return checkpoint_root / spec["phase"] / spec["id"] / "state.pt"


def make_initial(checkpoint_root: Path, model_config, shared: dict, config_digest: str) -> Path:
    """只生成一次随机初始模型，两个父分支都从这份权重出发。"""
    path = checkpoint_root / "initial.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config_digest"] != config_digest:
            raise RuntimeError("Existing initial checkpoint belongs to a different configuration")
        return path
    set_seed(int(shared["model_seed"]))
    model = TransformerLM(model_config)
    state = {
        "model": model.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config_digest": config_digest,
        "model_seed": int(shared["model_seed"]),
    }
    save_checkpoint(path, state)
    return path


def load_training_state(
    spec: dict, shared: dict, model_config, checkpoint_root: Path,
    config_digest: str, device: torch.device,
):
    """恢复本阶段最新进度；新子分支则从同一父检查点完整分叉。"""
    own_path = checkpoint_path(checkpoint_root, spec)
    if own_path.exists():
        source_path = own_path
    elif spec["phase"] == "stage1":
        source_path = checkpoint_root / "initial.pt"
    else:
        parent_spec = {"phase": "stage1", "id": spec["parent"]}
        source_path = checkpoint_path(checkpoint_root, parent_spec)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source["config_digest"] != config_digest:
        raise RuntimeError(f"Checkpoint config mismatch: {source_path}")
    if spec["phase"] == "stage2" and not own_path.exists():
        if source["stage_step"] != int(shared["stage_steps"]):
            raise RuntimeError(f"Parent has not reached fork: {source_path}")
        if source["run_id"] != spec["parent"]:
            raise RuntimeError("Wrong parent checkpoint")
    model = TransformerLM(model_config).to(device)
    model.load_state_dict(source["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(shared["peak_learning_rate"]),
        betas=(float(shared["beta1"]), float(shared["beta2"])),
        weight_decay=float(shared["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if "optimizer" in source:
        optimizer.load_state_dict(source["optimizer"])
        scaler.load_state_dict(source["scaler"])
    data_rng = np.random.default_rng(int(shared["data_seed"]))
    if "data_rng_state" in source:
        data_rng.bit_generator.state = source["data_rng_state"]
    # 模型构造会消耗随机数，所以最后再恢复训练时的 RNG 状态。
    torch.set_rng_state(source["cpu_rng"])
    if device.type == "cuda" and source["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(source["cuda_rng"])
    if source_path == own_path:
        stage_step = int(source["stage_step"])
        history = source["history"]
    elif spec["phase"] == "stage2":
        stage_step = 0
        history = list(source["history"])
    else:
        stage_step = 0
        history = []
    global_step = int(source.get("global_step", 0))
    data_chain = bytes.fromhex(source.get("data_chain", "00" * 32))
    skipped = int(source.get("skipped_updates", 0))
    del source
    return model, optimizer, scaler, data_rng, stage_step, global_step, history, data_chain, skipped


def pack_state(
    spec: dict, model, optimizer, scaler, data_rng, stage_step: int, global_step: int,
    history: list[dict], data_chain: bytes, skipped: int, config_digest: str,
) -> dict:
    """检查点包含从当前批次继续训练所需的完整状态。"""
    return {
        "run_id": spec["id"],
        "phase": spec["phase"],
        "train_context": spec["context_length"],
        "stage_step": stage_step,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "data_rng_state": data_rng.bit_generator.state,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": history,
        "data_chain": data_chain.hex(),
        "skipped_updates": skipped,
        "config_digest": config_digest,
    }


def write_status(root: Path, spec: dict, stage_step: int, global_step: int, stage_steps: int, message: str):
    """每次记录进度时更新一个简短状态文件，便于长训练期间查看。"""
    save_json(root / "status.json", {
        "current_run": spec["id"],
        "phase": spec["phase"],
        "stage_step": stage_step,
        "stage_steps": stage_steps,
        "global_step": global_step,
        "message": message,
        "updated_local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


def run_stage(
    spec: dict, shared: dict, model_config, data_config, base_config: dict,
    manifest: dict, train_tokens: np.memmap, val_tokens: np.memmap, val_starts: np.ndarray,
    prompts: list[dict], tokenizer, device: torch.device, results_root: Path,
    checkpoint_root: Path,
):
    """训练一个父阶段或子阶段；中断后从本阶段最近检查点继续。"""
    stage_steps = int(shared["stage_steps"])
    max_context = int(shared["model_max_context"])
    base_batch = int(shared["base_windows_per_micro"])
    accumulation = int(shared["gradient_accumulation"])
    train_start = int(data_config.train_english_tokens) if int(data_config.train_english_tokens) > 0 else 0
    tokens_per_step = base_batch * max_context * accumulation
    result_dir = results_root / spec["phase"] / spec["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "effective_config.json", {
        "shared": shared, "stage": spec, "base": base_config, "manifest": manifest,
    })
    model, optimizer, scaler, data_rng, stage_step, global_step, history, data_chain, skipped = load_training_state(
        spec, shared, model_config, checkpoint_root, manifest["config_digest"], device,
    )
    metrics_path = result_dir / "metrics.jsonl"
    write_history(metrics_path, history)
    update_comparison(results_root)
    writer = SummaryWriter(str(results_root / "tensorboard" / spec["id"]))
    for event in history:
        add_tensorboard(writer, event)
    model.train()
    started = time.perf_counter()
    start_stage_step = stage_step

    if stage_step == 0 and spec["phase"] == "stage1" and not history:
        metrics = evaluate(model, val_tokens, val_starts, shared, device)
        history.append(metric_event("stage1", spec["id"], spec["context_length"], 0, 0, metrics))
        write_history(metrics_path, history)
        add_tensorboard(writer, history[-1])
        update_comparison(results_root)
        print(f"{spec['id']} initial nll128={history[-1]['nll_128']:.4f} nll512={history[-1]['nll_512']:.4f}", flush=True)

    try:
        for next_stage_step in range(stage_step + 1, stage_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(accumulation):
                starts = data_rng.integers(
                    train_start,
                    len(train_tokens) - max_context,
                    size=base_batch,
                    dtype=np.int64,
                )
                windows = read_windows(train_tokens, starts, max_context)
                inputs_np, targets_np = split_windows(windows, spec["context_length"], max_context)
                data_chain = hashlib.sha256(
                    data_chain + starts.tobytes() + targets_np.tobytes()
                ).digest()
                inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
                targets = torch.tensor(targets_np, dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    _, loss = model(inputs, targets)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                losses.append(float(loss.detach()))
            lr = learning_rate(global_step, 2 * stage_steps, shared)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(shared["grad_clip"]))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
            stage_step = next_stage_step
            global_step += 1

            if stage_step == 1 or stage_step % int(shared["log_interval"]) == 0 or stage_step == stage_steps:
                elapsed = max(1e-6, time.perf_counter() - started)
                event = {
                    "kind": "training",
                    "stage": spec["phase"],
                    "run_id": spec["id"],
                    "train_context": spec["context_length"],
                    "stage_step": stage_step,
                    "global_step": global_step,
                    "tokens_seen": global_step * tokens_per_step,
                    "train_loss": float(np.mean(losses)),
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": (stage_step - start_stage_step) * tokens_per_step / elapsed,
                    "skipped_updates": skipped,
                }
                history.append(event)
                add_tensorboard(writer, event)
                print(
                    f"{spec['id']} {stage_step}/{stage_steps} "
                    f"loss={event['train_loss']:.4f} lr={lr:.7f} "
                    f"speed={event['tokens_per_second']:.0f} tok/s skipped={skipped}",
                    flush=True,
                )
                write_status(results_root, spec, stage_step, global_step, stage_steps, "training")

            early = spec["phase"] == "stage2" and stage_step in {1, 10, 50, 100, 250}
            should_eval = early or stage_step % int(shared["eval_interval"]) == 0 or stage_step == stage_steps
            if should_eval:
                metrics = evaluate(model, val_tokens, val_starts, shared, device)
                event = metric_event(
                    spec["phase"], spec["id"], spec["context_length"], stage_step, global_step, metrics
                )
                history.append(event)
                add_tensorboard(writer, event)
                writer.flush()
                write_history(metrics_path, history)
                update_comparison(results_root)
                print(
                    f"{spec['id']} validation step={global_step} "
                    f"L128={event['nll_128']:.4f} L256={event['nll_256']:.4f} "
                    f"L512={event['nll_512']:.4f} G={event['long_context_gain']:.4f}",
                    flush=True,
                )
                write_status(results_root, spec, stage_step, global_step, stage_steps, "validation")

            should_checkpoint = stage_step % int(shared["checkpoint_interval"]) == 0 or stage_step == stage_steps
            if should_checkpoint:
                save_checkpoint(
                    checkpoint_path(checkpoint_root, spec),
                    pack_state(
                        spec, model, optimizer, scaler, data_rng, stage_step, global_step,
                        history, data_chain, skipped, manifest["config_digest"],
                    ),
                )
                print(f"{spec['id']} checkpoint step={global_step}", flush=True)
    finally:
        writer.close()

    # 最终检查点已经先于生成保存，因此采样 RNG 不会影响子分支的分叉状态。
    samples_path = result_dir / "samples.json"
    if stage_step == stage_steps and not samples_path.exists():
        save_json(samples_path, {
            "run_id": spec["id"],
            "global_step": global_step,
            "samples": generate_samples(model, prompts, tokenizer, shared, device),
        })
    write_status(results_root, spec, stage_step, global_step, stage_steps, "complete")
    print(f"{spec['id']} COMPLETE data_chain={data_chain.hex()}", flush=True)
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()


def verify_chains(checkpoint_root: Path, config_digest: str):
    """父分支和四个子分支应看到同一串目标 token；摘要不等就停止解释。"""
    groups = (
        (("stage1", "parent128"), ("stage1", "parent512")),
        (("stage2", "128-128"), ("stage2", "128-512"), ("stage2", "512-512"), ("stage2", "512-128")),
    )
    for group in groups:
        chains = {}
        for phase, run_id in group:
            path = checkpoint_root / phase / run_id / "state.pt"
            if not path.exists():
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["config_digest"] != config_digest:
                raise RuntimeError(f"Config mismatch in {path}")
            chains[run_id] = state["data_chain"]
            del state
        if len(chains) == len(group) and len(set(chains.values())) != 1:
            raise RuntimeError(f"Controlled data streams differ: {chains}")
    return True


def main():
    """读取配置、核对数据身份，并按两个父阶段加四个子阶段顺序执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", default="configs/task_b/shared.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage-steps", type=int, default=None)
    parser.add_argument("--generation-new-tokens", type=int, default=None)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    shared_path = Path(args.shared)
    shared = read_yaml(shared_path)
    if args.output_root is not None:
        shared["output_root"] = args.output_root
    if args.stage_steps is not None:
        shared["stage_steps"] = args.stage_steps
    if args.generation_new_tokens is not None:
        shared["generation_new_tokens"] = args.generation_new_tokens
    if int(shared["stage_steps"]) < 1:
        parser.error("stage-steps must be positive")
    results_root = Path(shared["output_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        update_comparison(results_root)
        print(f"Updated comparison CSV files in {results_root}")
        return

    stage_config_root = shared_path.parent / "task_b"
    if not (stage_config_root / STAGE_FILES[0]).exists():
        stage_config_root = shared_path.parent
    specs = [read_yaml(stage_config_root / relative) for relative in STAGE_FILES]
    model_config, data_config, train_config = load_config(shared["base_config"])
    if model_config.block_size != int(shared["model_max_context"]):
        raise ValueError("base model block_size must match model_max_context")
    if int(shared["base_windows_per_micro"]) * int(shared["model_max_context"]) * int(shared["gradient_accumulation"]) != 8192:
        raise ValueError("Expected exactly 8192 target tokens per optimizer step")
    if set(shared["evaluation_contexts"]) != {128, 256, 512}:
        raise ValueError("Expected evaluation contexts 128, 256 and 512")
    if any(spec["context_length"] not in {128, 512} for spec in specs):
        raise ValueError("Training context must be 128 or 512")
    base_config = config_as_dict(model_config, data_config, train_config)
    config_digest = experiment_config_digest(shared, specs, base_config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    train_path = Path(data_config.train_bin)
    val_path = Path(data_config.val_bin)
    tokenizer_path = Path(data_config.tokenizer_model)
    identity = {
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
    }
    manifest = {
        "config_digest": config_digest,
        "data_identity": identity,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "tokens_per_step": 8192,
        "tokens_per_stage": int(shared["stage_steps"]) * 8192,
        "tokens_per_full_path": int(shared["stage_steps"]) * 2 * 8192,
        "total_experiment_tokens": int(shared["stage_steps"]) * 6 * 8192,
    }
    manifest_path = results_root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config_digest"] != config_digest or old["data_identity"] != identity:
            raise RuntimeError("Existing result directory has different config or source data")
    else:
        save_json(manifest_path, manifest)
    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
    train_start = int(data_config.train_english_tokens) if int(data_config.train_english_tokens) > 0 else 0
    val_start = int(data_config.val_english_tokens) if int(data_config.val_english_tokens) > 0 else 0
    if len(train_tokens) - train_start <= int(shared["model_max_context"]):
        raise ValueError("Training region is shorter than one window")
    val_starts = np.linspace(
        val_start,
        len(val_tokens) - int(shared["model_max_context"]) - 1,
        int(shared["validation_windows"]),
        dtype=np.int64,
    )
    save_json(results_root / "validation_indices.json", {"start_indices": val_starts.tolist()})
    tokenizer = load_tokenizer(data_config.tokenizer_model)
    prompts = select_prompts(val_path.with_suffix(".txt"), tokenizer, shared)
    save_json(results_root / "prompt_sources.json", {"prompts": prompts})
    checkpoint_root = Path(shared.get("checkpoint_root", "checkpoints/task_b"))
    if args.output_root is not None:
        checkpoint_root = Path("checkpoints/task_b_pilot")
    make_initial(checkpoint_root, model_config, shared, config_digest)
    print(
        f"Task B device={device} steps_per_stage={shared['stage_steps']} "
        f"tokens_per_full_path={manifest['tokens_per_full_path']:,}",
        flush=True,
    )
    for spec in specs:
        if spec["phase"] == "stage2":
            verify_chains(checkpoint_root, config_digest)
        run_stage(
            spec, shared, model_config, data_config, base_config, manifest,
            train_tokens, val_tokens, val_starts, prompts, tokenizer, device,
            results_root, checkpoint_root,
        )
    verify_chains(checkpoint_root, config_digest)
    update_comparison(results_root)
    save_json(results_root / "completion.json", {
        "completed": True,
        "config_digest": config_digest,
        "data_chains_match": True,
        "completed_local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    print("TASK B ALL PATHS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
