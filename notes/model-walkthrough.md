# 模型与训练链路讲解

这份说明对应 `configs/base.yaml` 和 `src/` 的实际实现。它是一个从随机参数开始、预测下一个 token 的 decoder-only Transformer。训练语料不是问答指令，所以训练阶段属于小规模从零预训练，没有做 supervised fine-tuning 或 alignment。

## 1. Token 与训练样本

`src/tokenizer.py` 用 16,000 词表的 SentencePiece Unigram 模型把文本切为 token，再把每个 token 映射到整数 ID。训练文本末尾追加 EOS ID。训练集在 `data/processed/train.bin` 中以 `uint16` 储存；词表小于 65,536，所以每个 ID 占两个字节。

`src/dataset.py` 从连续 token 流中取 513 个 ID。前 512 个是 `input_ids`，后 512 个是 `targets`。因此位置 `t` 的目标是位置 `t+1` 的 token。一个 micro-batch 的两个张量都是 `[4,512]`。随机窗口可能跨越文档边界，EOS 可以让模型看到边界，但并不完全阻止跨文档上下文。

## 2. Embedding

`src/model.py` 中的 `nn.Embedding(16000, 512)` 是可训练查找表。输入 `[B,T]` 的 ID 被索引成 `[B,T,512]`。模型没有单独的绝对位置向量；位置关系由每层 attention 中的 RoPE 提供。

## 3. Attention

每层先将 `[B,T,512]` 经过 LayerNorm，再通过一个线性层一次生成 Q、K、V，整体形状 `[B,T,1536]`。切成三份后，每份重排为 `[B,8,T,64]`。这与三个独立的 512→512 线性层在数学上等价。

RoPE 在每个 head 的 Q 和 K 上按位置旋转成对维度，不旋转 V。每个位置的 query 与之前所有位置的 key 做点积，得到 `[B,8,T,T]` 分数矩阵。除以 `sqrt(64)` 控制分数尺度。上三角的未来位置被置为负无穷，所以 softmax 后权重为零；第 `t` 个 token 不能读取 `t+1` 及以后的位置。

对最后一个维度做 softmax 得到注意力权重，权重乘 V 得到每个 head 的输出。8 个 head 拼接回 `[B,T,512]`，再经输出投影。残差连接把结果加回 block 输入。`src/smoke_test.py` 会修改输入后半段并验证前半段 logits 不变，直接检查因果约束。

## 4. MLP

Attention 后再做 LayerNorm。MLP 将每个位置独立地从 512 维投影到 2048 维，经过 GELU 非线性，再投影回 512 维。它不在不同 token 之间交换信息；跨位置的信息交换发生在 attention。MLP 输出通过第二个残差连接加回。

上述 attention 与 MLP 组成一个 Transformer block，共重复 8 层。Pre-LN 指 LayerNorm 在两个子模块之前，这有利于小模型稳定训练。

## 5. Logits 与 Loss

最后的 LayerNorm 后，输出层把每个位置的 512 维向量映射到 16,000 个词表分数，形状 `[B,T,16000]`。这些是 logits，不是概率。输出层和输入 embedding 共用同一权重矩阵，减少参数量。

`F.cross_entropy` 内部做稳定的 log-softmax，并取正确目标 token 的负对数概率。所有 `B*T` 个位置的损失求平均，得到一个标量。训练时无需显式把 logits 转成概率；生成时才用 softmax 将最后一个位置的 logits 转为采样概率。

## 6. Backward 与 Optimizer

`src/train.py` 中每次 forward 产生 loss，随后 `backward()` 沿计算图把损失对各参数的梯度算出来。FP16 autocast 减少计算和显存开销；GradScaler 放大 loss 以避免 FP16 小梯度下溢，更新前再还原梯度。每 4 个 micro-batch 累积一次梯度，共 `4 × 4 × 512 = 8192` 个 token，然后裁剪梯度范数并执行一次 AdamW 更新。

AdamW 为梯度的一阶矩和二阶矩维护指数移动平均，再按学习率更新参数，并单独执行权重衰减。学习率先 warmup 500 步，再沿余弦曲线从 `3e-4` 降到 `3e-5`。当前实现对所有参数统一做 weight decay，包括 bias 和 LayerNorm 参数；这是一个可以在后续复盘中讨论的简化选择。

## 7. 训练、验证与生成的区别

训练时启用 dropout、反向传播和参数更新。验证时关闭 dropout，不求梯度，在英文和中文验证区间各选固定位置的窗口，分别计算 loss，再以 65%/35% 加权。生成时只使用 prompt 和模型预测，不提供目标 token；每次取最后一个位置的 logits，按 temperature 和 top-k 调整后抽样一个新 token，并重复此过程。

训练用了约 2.4 亿 seen tokens，但训练文件只有约 1.13 亿 token，所以窗口被多次抽样。前约 5,700 万 seen tokens 使用原始英文 37.6% 的比例，之后采用英文 74% 的分层采样，以使累计训练比例接近 65% 英文、35% 中文。训练中途更换了采样和验证口径，不能把所有阶段的 loss 当作同一口径的连续实验。

## 8. 当前结果与边界

最终 checkpoint 为 `checkpoints/final_step_00029297.pt`，模型有 33,412,096 个参数。每种语言各 100 个固定验证 batch 的独立评估结果：英文 loss 1.64872、中文 loss 4.38744、65%/35% 加权 loss 2.60727。训练后段指标改善已明显放慢，但并未完全停止。

英文输出能形成简短故事和对话，仍可能重复或自相矛盾。中文输出有局部通顺词句，但整体语义仍不可靠。两种语料的文体差异很大，不能仅凭两种语言的 loss 数字断定模型“更懂”哪种语言。Tokenizer 的字符覆盖率设为 99.95%，没有 byte fallback，少见字符可能显示为未知 token。
