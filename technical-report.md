# 小型 Transformer 训练与实验技术总报告

## 目录与报告大纲

1. [项目目标和正式方案](#项目目标和正式方案)
2. [模型和训练链路](#模型和训练链路)
3. [数据方案、历史留存与实验隔离](#数据方案历史留存与实验隔离)
4. [基础训练结果](#基础训练结果)
5. [任务 A：KV cache 推理优化](#任务-a-kv-cache-推理优化)
6. [任务 B：训练 context 切换 dynamics](#任务-b训练-context-切换-dynamics)
7. [真实问题、定位和修复](#真实问题定位和修复)
8. [失败分析、限制和仍未知的内容](#失败分析限制和仍未知的内容)
9. [技术复核准备](#技术复核准备)

本报告按“先定义系统，再说明控制条件，再给结果，最后说明失败和边界”的顺序组织。这样每个结论都能回到配置、日志、checkpoint 或固定数据，而不是只依赖生成样例。英文 Task B 的专门计划文档是在实验后整理的；更早的旧中文计划保留了研究问题和四路径设计，不能把英文文档称为预注册。

## 项目目标和正式方案

目标是从零训练并解释一个能在 RTX 4060 Laptop GPU 上运行的小型 decoder-only Transformer，完整说明 token → embedding → attention → MLP → logits → loss → backward → optimizer。当前正式方案是只使用英文 TinyStories 的单语模型；此前的中英混合方案完整保留为历史记录。

正式模型仍为 8 层、hidden 512、8 attention heads、head dimension 64、MLP hidden 2048、RoPE、Pre-LN、GELU、最大 context 512、约 33.4M 参数、16k SentencePiece Unigram tokenizer。输入 embedding 和输出 lm_head 权重共享。

## 模型和训练链路

文本先由 SentencePiece 映射成 token ID。训练输入是 `[B,T]`，目标是向右移动一个 token 的 `[B,T]`。embedding 查表得到 `[B,T,512]`。每个 block 先 LayerNorm，再产生 Q/K/V，拆成 8 个 head，RoPE 作用于 Q/K，causal mask 禁止看未来，scaled dot-product attention 加权 V 后合并并投影。残差后进入第二个 LayerNorm 和 512→2048→512 的 GELU MLP，再做残差。

最后 LayerNorm 后用共享输出权重得到 `[B,T,16000]` logits。cross entropy 比较正确的下一个 token，`backward()` 计算梯度；梯度累积 4 个 micro-batch 后，AdamW 使用梯度、动量和 weight decay 更新参数。FP16 AMP、梯度裁剪、周期 checkpoint 和固定验证集用于控制训练。

主要实现入口是 `src/model.py`、`src/dataset.py`、`src/train.py`、`src/sample.py` 和 `src/task_b_experiment.py`。

## 数据方案、历史留存与实验隔离

### 旧中英方案如何留存

旧方案的全部内容仍保留原位：

- `data/processed/` 和旧 Hugging Face 缓存中的英文 TinyStories、中文 Wikipedia 数据；
- `tokenizer/spm.model`、`tokenizer/spm.vocab`；
- 根 `checkpoints/`、`checkpoints/task_b/` 和旧 `experiments/task_b/`；
- 旧 TensorBoard、JSONL、Task A KV cache 记录、Task B 中间结果和失败记录；
- 迁移说明 `notes/english-single-migration.md` 和原始 debug log。

这些目录没有被删除或覆盖，也没有被新配置引用。它们只用于历史复盘，不能作为新模型初始化、验证或新报告的结果来源。

### 新英文方案如何隔离

- 英文文本和 token 文件：`data/english_single/processed/`。
- 英文 tokenizer：`tokenizer/english_single/`，由英文训练文本重新训练。
- 基础训练配置、状态、日志：`configs/english_single/base.yaml`、`checkpoints/english_single/base/`、`experiments/english_single/base/`。
- Task A：`experiments/english_single/task_a/`，只加载英文基础训练最终 checkpoint。
- Task B：`configs/english_single/task_b/`、`checkpoints/english_single/task_b/`、`experiments/english_single/task_b/`。

`data/english_single/manifest.json` 和 Task B manifest 固定 train/val/tokenizer 的 SHA-256。新模型从随机初始化开始，optimizer、GradScaler、RNG、训练步数和验证指标均在新目录建立。新配置仍有为兼容共用代码而存在的语言边界字段，但 `train_english_tokens: 0`、`val_english_tokens: 0` 与 `english_sample_probability: 1.0` 使整个新文件按英文单语处理，不读取旧中文区域。因此旧 token ID、旧 embedding/output 行和旧优化器状态都不会进入新实验。

## 基础训练结果

英文 TinyStories 训练文档 196,702 篇、验证文档 3,947 篇；训练 42,074,460 token、验证 842,751 token。基础训练累计 240,001,024 seen tokens，最终 checkpoint 为 `checkpoints/english_single/base/final_step_00029297.pt`。训练日志最后一次验证发生在第 29,250 步，英文 loss 约 1.5810，perplexity 约 4.8599；这不是对第 29,297 步最终权重单独重新评估的数字。

训练曾在第 6000 步暂停，恢复时触发 CUDA RNG 类型问题；修复后从新英文 checkpoint 正常恢复并完成。这个过程作为真实 debug 记录保存在 `notes/debug-log.md`。

## 任务 A：KV cache 推理优化

问题是逐 token 生成时复用各层历史 K/V 是否减少重复计算，同时保持输出一致。固定英文单语最终 checkpoint、FP32、batch 1、生成 128 token、5 次计时：

| prompt token | 无缓存中位耗时 | KV cache 中位耗时 | 加速 |
|---:|---:|---:|---:|
| 32 | 1.671 s | 1.636 s | 1.02x |
| 256 | 1.534 s | 1.498 s | 1.02x |

贪心 token、固定 seed 随机采样和跨 512 context 上限重建的输出都一致；增量 logits 最大绝对差约 `4.77e-6`。由于模型小、生成长度短，Python 和 GPU 调度开销占比高，实测收益很小，但正确性和轻微速度收益均被记录。详细数据在 `experiments/english_single/task_a/kv-cache-results.json`。

交付审计时发现：2026-11-14 在已完成预算的训练入口执行后，最终 checkpoint 被重新保存；没有新的优化器更新，但文件 SHA-256 与 Task A 原始记录不再相同。原始哈希和计时保留，当前文件不能用于原始字节级哈希复核。具体数值和原因见 Task A 专门报告；训练入口已修正为完成后不覆盖文件。

## 任务 B：训练 context 切换 dynamics

完整问题、假设、设计和结果见 `experiments/english_single/task_b/task-b-report.md`。两条父路径从同一随机初始化开始；同一父路径的两个子分支再从完全相同的父状态分叉。父阶段各 80,003,072 token，第二阶段再训练同样预算。主要指标是固定英文目标上的 `L128`、`L256`、`L512` 和 `G=L128-L512`。

最终结果如下：

| 路径 | L128 | L512 | G |
|---|---:|---:|---:|
| 128-128 | 1.7874 | 3.6949 | -1.9074 |
| 128-512 | 1.7919 | 1.6099 | 0.1819 |
| 512-512 | 1.7931 | 1.5944 | 0.1986 |
| 512-128 | 1.7852 | 1.6315 | 0.1537 |

结论是：128→512 能建立长历史能力；512→128 仍改善短历史预测，但长历史收益下降；只看 L128 会错误地把 128-128 或 512-128 当成更好，因此必须同时报告多种评估 context。四条路径 data chain 一致，结果可复核。

## 真实问题、定位和修复

### CUDA RNG 恢复失败

症状是英文基础训练恢复时在第一批数据前报 `RNG state must be a torch.ByteTensor`。检查 checkpoint、CUDA RNG 和 `map_location` 后发现保存的 ByteTensor 被映射到 GPU，而 `torch.cuda.set_rng_state_all` 要求 CPU ByteTensor。修复为恢复前调用 `.cpu()`，随后训练完成。

### 英文 Task B 仍寻找中文 prompt

首次启动 Task B 在写 checkpoint 前报 `Not enough long Chinese documents`。检查英文验证文件和 prompt selector 后发现旧逻辑硬编码中文 Unicode 范围。改为检测到中文时使用中文，否则选择足够长英文文档；失败启动没有产生新 checkpoint。

这两件事都写入 `notes/debug-log.md`，包括症状、检查、根因、修复和下次应先做的检查。发现、定位和修复由 Codex 协助完成；技术复核时不能说成使用者独立完成。

## 失败分析、限制和仍未知的内容

旧方案的字符预算没有得到预期 token 比例，这是迁移纯英文方案的直接原因：英文和中文 token 化效率及文体差异造成了实验混杂。这个失败被保留为历史设计复盘，旧指标没有并入新结论。

新 Task B 只有一个随机种子、一个 TinyStories 划分和一个模型规模，不能声称结论普遍适用于所有模型。`G` 还混合了可见历史、RoPE 位置和窗口边界效应；没有自动文本质量指标；也没有测试多个切换点或多次切换。英文计划文档在实验完成后整理，不能作为英文假设事前登记的证据。报告初稿还误把父阶段第 5,000 步的数据放到子阶段“中点”；已根据 `comparison.csv` 更正为全局第 14,766 步。下一步若继续研究，应先增加随机种子或切换点，而不是只延长生成样例。

## 技术复核准备

应能现场说明：为什么 `L512` 不是简单把 prompt 变长、为什么固定最后 128 个 target、为什么要比较 `G` 和原始 loss、父阶段如何保证相同状态、KV cache 为什么不改变 logits、checkpoint 为什么必须包含 optimizer 和 RNG，以及为什么旧 tokenizer 不能加载新 embedding。现场修改 context 时，应同步说明它改变的是每个 target 的可见历史和 RoPE 位置，而不是 attention head 数量。
