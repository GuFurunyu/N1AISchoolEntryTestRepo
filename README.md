# 小型 Transformer 语言模型

这是一个从零训练的中英双语 decoder-only Transformer。模型规格见 `模型方案.txt`，目录说明见 `项目结构.txt`。源码在 `src/`，关键张量操作都有注释；运行时、依赖、数据缓存、tokenizer、checkpoint 和训练日志均位于项目目录内。

## 当前状态

- Python 3.12 运行时：`.runtime/python/`
- 独立环境：`.venv/`
- CUDA PyTorch：`2.5.1+cu124`
- 模型参数：33,412,096
- 语料：训练集 112,968,440 tokens；验证集 2,401,669 tokens
- 数据源：TinyStories 英文 + 中文 Wikipedia
- tokenizer：16k SentencePiece Unigram
- GPU smoke test、真实数据 forward/backward、pilot 训练、checkpoint 恢复均已通过
- 正式训练已完成：step 29,297，累计 240,001,024 seen tokens；最终 checkpoint：`checkpoints/final_step_00029297.pt`
- 最终独立评估（中英文各 100 batch）：英文 loss 1.64872，中文 loss 4.38744，加权 loss 2.60727

逐层结构、张量形状和训练链路见 `notes/model-walkthrough.md`。训练后段验证 loss 下降明显放缓，但尚未完全停止。英文生成有故事结构，中文生成仍缺乏可靠的整体语义。

## 架构与张量路径

输入文本先由 SentencePiece 切成 token，再映射成整数 ID。训练 batch 的输入 `input_ids` 和目标 `targets` 形状都是 `[B,T]`；目标是输入向右移动一个 token。

1. `token_embedding` 查表，将 `[B,T]` 的 ID 变成 `[B,T,512]` 的向量。每个 token 的 512 个值是模型可训练参数。
2. 每个 Transformer block 先做 LayerNorm，再用一个线性层产生 Q、K、V；它们被拆成 8 个 head，每个 head 维度 64。RoPE 旋转 Q 和 K 来提供位置信息。
3. Attention 计算 `Q @ K^T / sqrt(64)`，把未来位置 mask 成负无穷，softmax 后对 V 加权求和。8 个 head 的输出拼接，再通过输出线性层投影回 512 维。
4. 残差连接将 attention 输出加回输入。第二个 LayerNorm 后，MLP 把 512 维扩展到 2048 维，经过 GELU，再压回 512 维；结果再次通过残差连接加回。
5. 8 个 block 后经过最后的 LayerNorm，用与输入 embedding 共享权重的输出层得到 `[B,T,16000]` logits。每个位置的 16000 个值是下一个 token 的未归一化分数。
6. Cross entropy 将 logits 与正确的下一个 token ID 比较，得到标量 loss。`backward()` 通过计算图求梯度，AdamW 根据梯度、动量和权重衰减更新参数。梯度累积 4 个 micro-batch 后才更新一次。

主要代码入口：`src/model.py`（架构）、`src/dataset.py`（训练窗口）、`src/train.py`（loss/backward/AdamW/checkpoint）、`src/sample.py`（生成）。

## 训练控制

所有命令从项目根目录执行。

```powershell
# 测试模型 forward、loss、backward 和参数更新
.\.venv\Scripts\python.exe -m src.smoke_test

# 查看 GPU 速度和显存（使用随机 token）
.\.venv\Scripts\python.exe -m src.benchmark --steps 20

# 开始或自动恢复训练；当前目标已经完成，直接运行会立即结束
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train.ps1

# 指定训练目标（seen tokens，适合短试跑）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train.ps1 --max-tokens 5000000

# 查看训练曲线（浏览器访问 http://localhost:6006）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\view_logs.ps1

# 在终端看最近日志
Get-Content runs\train.jsonl -Tail 20

# 检查 GPU 利用率和显存
nvidia-smi
```

训练中按 `Ctrl+C` 可停止。程序会尝试保存中断 checkpoint；若终端强制结束进程，使用最近的周期 checkpoint 恢复。`--resume auto` 默认选取步数最大的 checkpoint，`--resume none` 从随机初始化开始。checkpoint 保存在 `checkpoints/`，包括模型、AdamW 状态、GradScaler 状态、步数和配置。不要同时启动两个训练进程写入同一个 `checkpoints/` 目录。

断点恢复会读回 PyTorch 随机数状态，但 DataLoader 的窗口缓冲未保存，因此恢复后的样本顺序不保证逐个与未中断运行相同。

训练参数在 `configs/base.yaml` 中修改。`micro_batch_size` 控制一次 forward/backward 的样本数；`grad_accum_steps` 控制多少次反向传播后更新一次；它们与 `block_size` 的乘积是每步的有效 token 数。修改模型结构、词表大小或 tokenizer 后，旧 checkpoint 通常不能直接恢复。

## 查看效果

```powershell
# 在 Windows 终端显示中文输出
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 分别报告英文、中文及按 65%/35% 加权的验证 loss
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint checkpoints\final_step_00029297.pt --batches 100

# 英文生成
.\.venv\Scripts\python.exe -m src.sample --checkpoint checkpoints\final_step_00029297.pt --prompt "Once upon a time" --max-new-tokens 120 --seed 1337

# 中文生成
.\.venv\Scripts\python.exe -m src.sample --checkpoint checkpoints\final_step_00029297.pt --prompt "从前有一个" --max-new-tokens 120 --seed 1337

# 检查语料的真实 token 比例
.\.venv\Scripts\python.exe -m src.data_stats --text data\processed\train.txt
```

验证口径曾变化：step 6000 之前主要评估文件开头的英文；step 6000 到 7000 采用全文件均匀抽样；step 7000 后分别评估中英文并按 65%/35% 加权。不同口径的总体 loss 不能直接相互比较，应在 `runs/train.jsonl` 中看 `val_sampling` 字段。真实问题和处理过程见 `notes/debug-log.md`。

## 重新准备数据

已下载的原始文件保存在 `data/cache/`。如需重新构建 tokenizer 和二进制 token 文件：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_data.ps1 --skip-download
```

这会重训 tokenizer；重新生成的 tokenizer 可能改变 token ID，不能与旧 checkpoint 混用。完整下载命令是 `scripts/prepare_data.ps1`，会从 Hugging Face 获取 TinyStories 和中文 Wikipedia。不要在训练进行时重建数据或 tokenizer。当前训练已完成，继续训练需要提高 `max_tokens`，但这也会改变余弦学习率时间轴；应把它记录为新的训练阶段。
