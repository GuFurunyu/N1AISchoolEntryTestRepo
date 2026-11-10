"""用随机 token 测量短时间训练吞吐量和显存峰值。

这不是语料训练：输入和目标都随机生成，loss 没有语言学习意义。
它只用于粗略估算本机速度、显存需求和长时间训练的可行性。
"""

import argparse
import time

import torch

from .config import load_config
from .model import TransformerLM


def main():
    """按配置创建模型，重复若干次前向、反向和优化器更新。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    model_config, _, train_config = load_config(args.config)
    # 没有 CUDA 时自动改用 CPU，此时测得的速度不能代表 GPU。
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    model.train()
    # 此基准只用 AdamW 的默认其他参数；不是完整训练脚本的配置复现。
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    # 形状与真实训练的小批一致，但 token 值从词表中随机抽取。
    inputs = torch.randint(0, model_config.vocab_size, (train_config.micro_batch_size, model_config.block_size), device=device)
    targets = torch.randint(0, model_config.vocab_size, inputs.shape, device=device)
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        # GPU 指令异步执行；计时前等待之前的任务完成。
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        # 一轮计时包含清梯度、前向求 loss、反向求梯度和参数更新。
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        # 再等 GPU 真正执行完，避免把尚未执行的计算漏出计时范围。
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = args.steps * inputs.numel()
    # 这里统计的是每轮一个小批的 token 数；没有梯度累积和数据加载开销。
    print(f"device={device}")
    print(f"steps={args.steps}")
    print(f"tokens={tokens:,}")
    print(f"tokens_per_second={tokens / max(elapsed, 1e-6):.2f}")
    if device.type == "cuda":
        # PyTorch 报告其分配器记录的峰值，不等于整块显卡的全部占用。
        print(f"peak_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.3f}")


if __name__ == "__main__":
    # 直接运行本模块时才进行基准测量。
    main()
