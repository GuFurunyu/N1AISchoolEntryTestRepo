"""从正式英文实验的 TensorBoard 事件和 Task A 原始 JSON 重画报告用图。

运行：.venv\\Scripts\\python.exe scripts\\plot_report_figures.py
图像写入 experiments/english_single/report_figures/；本脚本只读取实验记录。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Matplotlib 的字体等缓存也放在项目内，避免写入用户目录。
ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime" / "matplotlib")

import matplotlib

# 使用不需要打开窗口的绘图后端，便于在命令行直接保存 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EXPERIMENTS = ROOT / "experiments" / "english_single"
OUTPUT = EXPERIMENTS / "report_figures"
COLORS = {
    "128-128": "#d55e00",
    "128-512": "#0072b2",
    "512-512": "#009e73",
    "512-128": "#cc79a7",
}


def scalars(log_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """读取 TensorBoard 标量；目录读取会合并同一训练的续跑事件文件。"""
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    points = events.Scalars(tag)
    return [point.step for point in points], [point.value for point in points]


def save_figure(name: str) -> None:
    """保存高清图片，关闭窗口以免多张图相互叠加。"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT / name, dpi=220, bbox_inches="tight")
    plt.close()


def draw_base() -> None:
    """基础训练：训练损失和验证损失，以及单独的验证困惑度。"""
    log_dir = EXPERIMENTS / "base" / "tensorboard"
    train_steps, train_loss = scalars(log_dir, "train_loss")
    val_steps, val_loss = scalars(log_dir, "val_loss")
    _, perplexity = scalars(log_dir, "perplexity")

    plt.figure(figsize=(9, 4.5))
    plt.plot(train_steps, train_loss, color="#777777", alpha=0.45, linewidth=0.8,
             label="Training loss (each logged batch)")
    plt.plot(val_steps, val_loss, color="#0072b2", linewidth=2,
             label="Validation loss")
    plt.title("Base training: token prediction loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Cross-entropy loss (nats/token)")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure("base_loss.png")

    plt.figure(figsize=(9, 4.5))
    plt.plot(val_steps, perplexity, color="#009e73", linewidth=2)
    plt.title("Base training: validation perplexity")
    plt.xlabel("Optimizer step")
    plt.ylabel("Perplexity = exp(validation loss)")
    plt.grid(alpha=0.25)
    save_figure("base_perplexity.png")


def draw_task_a() -> None:
    """Task A 没有 TensorBoard 事件；用原始逐次计时展示波动范围。"""
    path = EXPERIMENTS / "task_a" / "kv-cache-results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, case in zip(axes, result["cases"]):
        trials = range(1, len(case["cached"]) + 1)
        ax.plot(trials, [run["seconds"] for run in case["uncached"]],
                marker="o", color="#d55e00", label="No cache")
        ax.plot(trials, [run["seconds"] for run in case["cached"]],
                marker="o", color="#0072b2", label="KV cache")
        ax.set_title(f"Prompt: {case['prompt_tokens']} tokens")
        ax.set_xlabel("Trial number")
        ax.set_xticks(list(trials))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Time to generate 128 tokens (seconds)")
    axes[0].legend()
    fig.suptitle("Task A: repeated inference timing")
    fig.tight_layout()
    save_figure("task_a_latency.png")


def draw_task_b() -> None:
    """Task B：四条子路径的事件文件已包含各自继承的父阶段曲线。"""
    log_root = EXPERIMENTS / "task_b" / "tensorboard"
    metrics = [
        ("validation/nll_context_128", "L128: 128-token evaluation"),
        ("validation/nll_context_512", "L512: 512-token evaluation"),
        ("validation/long_context_gain", "G = L128 - L512"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, (tag, title) in zip(axes, metrics):
        visible_values = []
        for name, color in COLORS.items():
            steps, values = scalars(log_root / name, tag)
            ax.plot(steps, values, color=color, linewidth=1.8, label=name)
            visible_values.extend(value for step, value in zip(steps, values) if step >= 7500)
        ax.axvline(9766, color="#333333", linestyle="--", linewidth=1,
                   label="Switch point" if ax is axes[0] else None)
        ax.set_ylabel("NLL (nats/token)" if "nll" in tag else "NLL difference")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xlim(7500, 19532)
        # 事件文件包含第 0 步约 9.76 的初始损失；当前图只展示分叉附近，
        # 因此按可见时间段定纵轴，避免早期极值把四条后期曲线压平。
        lower, upper = min(visible_values), max(visible_values)
        margin = max((upper - lower) * 0.08, 0.02)
        ax.set_ylim(lower - margin, upper + margin)
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Global optimizer step (dashed line: context switch)")
    fig.suptitle("Task B: training context switch and fixed-target validation", fontsize=13)
    fig.tight_layout()
    save_figure("task_b_dynamics.png")


if __name__ == "__main__":
    draw_base()
    draw_task_a()
    draw_task_b()
    print(f"Report figures saved in: {OUTPUT}")
