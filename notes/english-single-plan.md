# 英文单语实验计划

## 目的

重新建立一个只使用 TinyStories 英文的训练链路，消除旧中英混合方案中语言、文体、词汇复杂度和 token 化效率的混杂。旧结果不作为新模型的初始化、验证或结论。

## 数据和模型

- 数据：`roneneldan/TinyStories` 的 train split，按文档随机抽取 2% 作为验证集。
- 实际文件：训练 `42,074,460` tokens，验证 `842,751` tokens。
- tokenizer：只用新训练文本训练 16,000 词 SentencePiece Unigram，路径为 `tokenizer/english_single/`。
- 模型：8 层、512 hidden、8 heads、512 最大上下文，约 33.4M 参数，与旧结构相同但 token ID 和权重全部重新建立。
- 基础训练：随机初始化、AdamW、FP16、240M seen tokens；输出到 `checkpoints/english_single/base/`。

## 实验隔离

新基础训练使用 `configs/english_single/base.yaml`。Task A 使用新基础训练最终 checkpoint，结果写入 `experiments/english_single/task_a/`。Task B 使用 `configs/english_single/task_b_shared.yaml` 和其下四条阶段配置，检查点和结果分别写入 `checkpoints/english_single/task_b/` 与 `experiments/english_single/task_b/`。

新配置的 `train_english_tokens: 0`、`val_english_tokens: 0` 是单语标记：训练窗口从整个英文文件随机抽取，验证窗口从整个英文验证文件均匀抽取，不存在中文区域或语言采样概率。所有新报告必须记录配置、文件哈希、tokenizer 哈希和 checkpoint 哈希。
