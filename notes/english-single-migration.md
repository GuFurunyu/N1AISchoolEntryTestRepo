# 英文单语方案迁移记录

## 旧方案冻结

旧中英混合方案使用 TinyStories 英文、中文 Wikipedia、旧 SentencePiece、旧 `data/processed/`、旧 `tokenizer/spm.model`、旧 `checkpoints/`、旧 `experiments/task_b/` 和旧 `notes/kv-cache-*`。这些路径和文件作为历史记录保留，不删除、不覆盖，也不作为新实验的初始化、验证或结论来源。

## 新方案隔离

- 文本和 token 文件：`data/english_single/processed/`
- Hugging Face 缓存：只读取 TinyStories；新生成的处理文件不会写入旧 `data/processed/`
- tokenizer：`tokenizer/english_single/spm.model` 和 `.vocab`
- 基础训练配置、输出和 TensorBoard：`configs/english_single/base.yaml`、`checkpoints/english_single/base/`、`experiments/english_single/base/`
- Task B 配置、输出和完整状态：`configs/english_single/task_b/`、`checkpoints/english_single/task_b/`、`experiments/english_single/task_b/`
- Task A 结果：`experiments/english_single/task_a/`，使用英文单语基础训练最终 checkpoint

新 tokenizer 会产生新的 token ID 映射；新模型从随机初始化开始，AdamW、GradScaler、PyTorch RNG、训练步数和验证指标均来自新配置。新配置中的 `train_english_tokens: 0` 和 `val_english_tokens: 0` 表示整个文件都是英文，不启用旧双语语言边界或采样概率。

## 可复核边界

每个新实验目录保存配置快照、数据/ tokenizer SHA-256、checkpoint 路径和 JSONL/TensorBoard 记录。新报告只引用 `experiments/english_single/` 下的结果；旧结果只在迁移说明和失败复盘中作为背景提及。
