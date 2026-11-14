# 英文单语 Task A：KV cache 对照实验

## 问题

在同一个英文单语 Transformer、同一个英文 tokenizer 和同一个基础训练 checkpoint 下，逐 token 生成时复用各层历史 Key/Value，是否能减少重复计算并保持输出一致？

## 控制条件

- checkpoint：`checkpoints/english_single/base/final_step_00029297.pt`
- tokenizer：`tokenizer/english_single/spm.model`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- batch size：1；`torch.inference_mode()`；FP32 权重推理
- 每个 prompt 生成 128 个 token，temperature 1.0、top-k 1；5 次正式计时，缓存和无缓存交替作为先后顺序
- 32 和 256 token 的英文 prompt；只使用英文，避免把旧中英方案的中文 prompt 带入新实验

## 结果

| prompt token 数 | 无缓存中位耗时 | KV cache 中位耗时 | speedup | 输出是否一致 |
|---:|---:|---:|---:|:---:|
| 32 | 1.671 s | 1.636 s | 1.02x | 是 |
| 256 | 1.534 s | 1.498 s | 1.02x | 是 |

prefill logits 的最大绝对差为 0；增量 logits 最大绝对差约 `4.77e-6`，下一 token 的贪心 ID 相同。固定 seed 的 top-k=50 随机采样输出也相同；跨过 512 token 上限的缓存重建检查也相同。逐次原始数据、checkpoint 哈希和显存字段见 `kv-cache-results.json`。

### checkpoint 文件哈希的后续变化

2026-11-14 交付审计时，训练入口在已达到 29,297 步的情况下读取最终 checkpoint，未执行新的训练步骤，但旧代码仍在函数末尾把状态重新保存到同名文件。当前文件 SHA-256 为 `75D058EB014E3489F26B0011B1A846CF764FDBB93E855D3595B8B80079361F82`，与 Task A 原始记录中的 `5197FC8EB7B7A3408FB41D01E5356EE49824723E2DB68AFBD744E539B2E0CA71` 不同。原始逐次测量与原始哈希均保留，不把当前文件冒充为当时字节完全相同的文件；原始文件已无法用于逐字节复核。训练入口现已改为达到预算后直接退出，避免再次覆盖。

## 解释和限制

这次模型很小、生成长度只有 128，单 token 的 Python、采样和 GPU 调度开销占比高，因此实测加速只有约 1.02 倍；这不否定 KV cache 对长上下文和更长生成的理论收益。实验只测一个 checkpoint、一个 GPU 和两个英文 prompt 长度，不能推广到所有模型或服务负载。结果证明缓存路径在本模型上保持数值和输出行为一致，并在本测量条件下有轻微速度收益。
