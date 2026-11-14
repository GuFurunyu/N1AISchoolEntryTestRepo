# 当前英文单语模型与训练链路讲解

本文对应 `configs/english_single/base.yaml` 和 `src/` 的实际实现。模型从随机参数开始，学习预测下一个英文 TinyStories token。它不是问答模型，也没有做指令微调。

## 1. Token 和训练样本

`tokenizer/english_single/spm.model` 是仅用英文训练文本训练的 16,000 词 SentencePiece Unigram tokenizer。`src/tokenizer.py` 把英文故事切成 token，再把 token 转成整数 ID。每篇训练文本后有 EOS 边界；`data/english_single/processed/train.bin` 以 `uint16` 保存 token ID，每个 ID 占两个字节。

`src/dataset.py` 从 token 流取连续的 513 个 ID：前 512 个成为输入 `input_ids`，后 512 个成为目标 `targets`。于是输入位置 `t` 预测原始序列中的下一个 ID。一个 micro-batch 的输入和目标形状都是 `[4,512]`。随机窗口可能跨越两篇故事；EOS 提供文档边界信号，但不会从 attention 中自动屏蔽另一篇故事。

## 2. Embedding

`nn.Embedding(16000, 512)` 是一张可训练的查找表。输入 `[B,T]` 的 ID 索引出 `[B,T,512]` 的向量。向量不是词义的人工编码，而是在预测任务的梯度更新中逐步学到的参数。没有额外的绝对位置 embedding；位置由 attention 内的 RoPE 表示。

## 3. 因果 Attention

每层先做 LayerNorm，再用一个线性层同时产生 Q、K、V，形状 `[B,T,1536]`。拆成三份并重排后，每份是 `[B,8,T,64]`：8 个 head、每个 head 64 维。RoPE 旋转 Q 和 K，V 不旋转。

每个位置的 Q 与所有位置的 K 做点积并除以 `sqrt(64)`，得到 `[B,8,T,T]` 的分数。因果 mask 把未来位置变成不可选，softmax 后的权重只能落在当前和过去位置。权重乘 V 得到各 head 的输出，拼接成 `[B,T,512]`，再经输出投影和残差相加。Attention 是不同 token 之间交换信息的部分。

## 4. MLP 与 8 个 Block

Attention 残差后再做 LayerNorm。MLP 对每个位置独立做 512→2048 线性变换、GELU、2048→512 线性变换，然后做第二次残差相加。它增加每个位置的非线性表达能力，不直接交换不同位置的信息。这样的 block 重复 8 次；Pre-LN 表示 LayerNorm 位于 attention 和 MLP 之前。

## 5. Logits 和 Loss

最后的 LayerNorm 后，共享输出权重把每个位置映射到 16,000 个词表分数，logits 形状为 `[B,T,16000]`。分数还不是概率。交叉熵内部做稳定的 log-softmax，再读取正确目标 ID 的负对数概率，最后对所有 `B×T` 位置求平均，得到标量 loss。输出层与输入 embedding 共享权重，减少参数量。

## 6. Backward 和 AdamW

`src/train.py` 用 FP16 autocast 计算前向。GradScaler 放大 loss，减少小梯度在 FP16 下溢的风险；`backward()` 沿计算图求参数梯度。每 4 个 micro-batch 累积一次梯度，即 `4×4×512=8192` 个目标 token，然后还原梯度、裁剪范数并执行一次 AdamW 更新。

AdamW 保存梯度的一阶、二阶移动平均，结合学习率和权重衰减更新参数。学习率先 warmup 500 步，再从 `3e-4` 按余弦日程降到 `3e-5`。当前实现对 bias 和 LayerNorm 参数也使用统一 weight decay，这是一个实现简化。checkpoint 包含模型、optimizer、GradScaler 和训练状态；断点恢复必须使用同一 tokenizer 与模型结构。

## 7. 训练、验证和生成

训练时启用 dropout、反向传播和权重更新。验证时关闭 dropout，不求梯度，使用 `data/english_single/processed/val.bin` 的固定英文窗口计算 loss 和 perplexity。生成时不给正确目标，只取最后位置 logits，按 temperature 和 top-k 选择新 token，再重复。可选 KV cache 在生成时复用各层 K/V，不改变已学权重。

## 8. 当前结果和边界

基础训练最终 checkpoint 是 `checkpoints/english_single/base/final_step_00029297.pt`，累计 240,001,024 seen tokens。最近记录的英文验证指标为第 29,250 步 loss 1.5810、perplexity 4.8599。TinyStories 生成可以形成简单故事，但重复与逻辑错误仍可能出现；验证 loss 不能单独证明故事质量。

任务 A 的正式英文 KV cache 结果在 `experiments/english_single/task_a/`。任务 B 是另一组独立随机初始化的训练路径，结果在 `experiments/english_single/task_b/`，不应把它的 loss 与基础模型直接当作同一训练轨迹。

旧中英模型的分语言验证、65%/35% 加权 loss 和中文生成结论属于历史方案，见 `notes/experiment-log.md`；它们不适用于当前英文 checkpoint。
