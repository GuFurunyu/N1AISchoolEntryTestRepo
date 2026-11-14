# 历史保留：旧中英模型的任务 A KV cache 实验

> 本文数据来自旧中英混合模型，不能用于当前英文单语模型的性能结论。
> 当前正式英文实验见 `experiments/english_single/task_a/kv-cache-report.md`。

## 问题与预期

问题：同一个训练好的 Transformer，在逐 token 生成时，复用历史 Key/Value（KV cache）会怎样改变输出、速度和显存？

预期：输出应与每轮重算完整上下文相同；上下文越长，避免重复计算带来的提速越明显；保存各层 K/V 会增加显存占用。短上下文下，缓存管理的额外开销可能抵消收益。这里研究的是推理实现，没有重新训练模型。

## 实现与对照

- 无缓存：沿用 `src/sample.py` 原生成路径。每生成一个 token，都将最近最多 512 个 token 送入模型，重新计算各层 Q/K/V 和 logits。
- 有缓存：在 `src/model.py` 增加 `forward_cached`。第一次处理完整提示词；之后每轮只把刚生成的 token 送入模型，并在每层拼接和复用之前的 K/V。新 token 的 RoPE 位置从已缓存长度开始。
- 当上下文超过模型的 512-token 上限时，有缓存路径从最近 512 个 token 重建缓存。这与原路径的截断方式一致，但超长生成时会失去大部分加速。
- 两条路径使用同一个训练检查点 `checkpoints/final_step_00029297.pt`（step 29297）、同一个 SentencePiece 分词器、batch=1 和 `torch.inference_mode()`。检查点 SHA-256 和全部逐次结果保存在 `kv-cache-results.json`。

性能测量在 RTX 4060 Laptop GPU、PyTorch 2.5.1+cu124 上进行。英文和中文各有 32、256 token 两种提示词长度；为了精确控制长度，固定句子被重复并截断。每次生成 128 个新 token，使用 `temperature=1.0, top_k=1`，因而选词是确定的。两条路径分别先完成一次同样长度的预热，再各测 5 次，并交替测量顺序。用 `torch.cuda.synchronize()` 确保 GPU 工作被计入计时。下表时间为 5 次的中位数，包含提示词处理、逐 token 解码与采样。

## 结果

| 语言 | 提示词 token | 无缓存时间 | 有缓存时间 | 加速比（无/有） | 无缓存峰值显存 | 有缓存峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 英文 | 32 | 1.021 s | 1.003 s | 1.02× | 151.8 MiB | 152.4 MiB |
| 英文 | 256 | 1.592 s | 0.993 s | 1.60× | 167.0 MiB | 176.3 MiB |
| 中文 | 32 | 1.097 s | 1.150 s | 0.95× | 151.8 MiB | 152.4 MiB |
| 中文 | 256 | 1.594 s | 1.085 s | 1.47× | 167.0 MiB | 176.3 MiB |

峰值显存是 PyTorch 的 `max_memory_allocated`，包含模型权重，不是整块 GPU 的全部占用。256-token 提示词下缓存峰值高约 9.3 MiB；这并不等于纯 K/V 大小，因为两条路径还有不同的临时张量。

四种条件下，两条路径的 128 个贪心生成 token 全部一致。完整提示词首次前向的最后一个位置 logits 最大绝对差为 0；追加一个 token 后的最大绝对差在约 `4.1e-6` 到 `1.3e-5` 之间，属于浮点运算顺序带来的微小差异。固定随机种子 1337、`temperature=0.8, top_k=50` 时，英文与中文自然提示词各生成 64 token，两条路径输出也一致。英文样例接续了儿童发现花园小门的故事；中文样例仍存在语义不连贯，两种路径生成的是同一段文字。提示词接近 512-token 上限并跨过上限时，重建缓存后的输出也一致。

## 实际遇到的测量问题

本次实现、运行和下面两项问题的定位由 Codex 协助完成。它们是实验过程中真实发生的事，但不应表述为使用者独立发现和解决；技术复盘前应亲自复跑并能解释修正依据。

初次实验把完整训练检查点用 `map_location=device` 载入 GPU。检查点不仅有模型权重，还含 AdamW 优化器状态；后者与推理无关。检查原始记录时，发现推理峰值显存约 536–570 MiB，与 3341 万参数模型的纯权重规模不相称。定位到检查点加载语句后，改为先在 CPU 读取、只将模型权重复制到 GPU，并删除临时状态；重测的峰值降到约 152–176 MiB。初次探索数据留在 `kv-cache-pilot.json`，其显存数值不能用于本报告结论。

随后注意到短英文提示词的初轮耗时偏高，而当时仅预热 8 token，正式却生成 128 token。于是把预热改为同样的 128 token，重复次数从 3 增至 5，再得出上表。这个修正降低了首次运行开销对中位数的影响，但短条件仍有明显波动，不能解读 1.02× 和 0.95× 为稳定的语言差异。

## 结论与限制

本机、本模型、128-token 生成条件下，KV cache 在 256-token 提示词上约提速 1.47–1.60 倍，代价是更高的 PyTorch 峰值显存；在 32-token 提示词上没有稳定提速。输出相同说明这次改动主要影响计算方式，而非模型学到的内容。

这不是通用的 KV cache 加速倍数：实验只有一台 GPU、batch=1、FP32、两种长度和少量重复；耗时包含采样及 Python 开销。当前实现每步用 `torch.cat` 扩展 K/V，没有预分配缓存，也没有高效 attention kernel。超过 512 token 时为保持与原模型一致而重建缓存；若改用滑动窗口丢弃旧 K/V，旧隐藏状态可能仍含被丢弃 token 的信息，不能直接声称与原路径等价。后续可分别测量 prefill 与 decode、使用更长输出及不同 batch，并比较预分配缓存，但这些不属于本次结论。

## 复现命令

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m src.inference_experiment --checkpoint checkpoints/final_step_00029297.pt --new-tokens 128 --trials 5
```

手动对比同一提示词的生成路径：

```powershell
.\.venv\Scripts\python.exe -m src.sample --checkpoint checkpoints/final_step_00029297.pt --prompt "Once upon a time" --max-new-tokens 64 --seed 1337
.\.venv\Scripts\python.exe -m src.sample --checkpoint checkpoints/final_step_00029297.pt --prompt "Once upon a time" --max-new-tokens 64 --seed 1337 --kv-cache
```
