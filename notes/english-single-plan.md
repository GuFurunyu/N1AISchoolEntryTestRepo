# 英文单语方案：设计与完成记录

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

新配置的 `train_english_tokens: 0`、`val_english_tokens: 0` 是单语标记：训练窗口从整个英文文件随机抽取，验证窗口从整个英文验证文件均匀抽取，不存在中文区域或按语言抽样。数据、tokenizer 和配置身份由各自 manifest 与配置文件复核；Task A 的原始 JSON 还记录基础 checkpoint 哈希。

## 实际完成情况

- 基础训练完成 29,297 步、240,001,024 seen tokens；最终状态为 `checkpoints/english_single/base/final_step_00029297.pt`。第 29,250 步英文验证 loss 1.5810、perplexity 4.8599。
- Task A 已用新英文模型完成 KV cache 对照；32 和 256 token 提示词均测得约 1.02 倍加速。报告在 `experiments/english_single/task_a/kv-cache-report.md`，原始 JSON 记录了 checkpoint SHA-256。
- Task B 已从另一个共同随机初始状态完成四条路径，`completion.json` 记录数据链一致。报告在 `experiments/english_single/task_b/task-b-report.md`；该目录的 manifest 记录数据、tokenizer 和配置身份，各阶段 checkpoint 保存在独立路径。
- 本文件是迁移设计和完成状态的说明。英文 Task B 的详细假设文档是在训练后整理的，不能将其称为英文实验的预注册。
