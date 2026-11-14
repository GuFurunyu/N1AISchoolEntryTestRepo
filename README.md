# 小型 Transformer 语言模型

这是一个用于完成N1 AI School考核的个人实验项目。项目从随机初始化开始训练一个可在个人 RTX 4060 Laptop GPU 上运行的小型 decoder-only Transformer，并记录模型实现、训练过程、调试问题和受控实验结果。

**发送考核材料时，只发送根目录下的 `submission_english_single/` 文件夹。** 它是当前英文单语方案的独立复核包，包含可运行的源码、数据、tokenizer、最终模型、Task B 状态和实验记录。请只压缩该文件夹，不要把外层完整项目一起压缩；包内的 `交付包说明.txt` 列出运行条件和复核入口。该大文件夹被 Git 忽略，因此仅提供 GitHub 仓库链接不足以交付模型和数据。原项目保留旧中英方案及周期 checkpoint，供本地历史复盘。

## 考核内容

考核要求分为两个主要任务：

- **任务 A：从零理解并训练模型**。需要能够解释 token、embedding、causal attention、MLP、logits、loss、反向传播和优化器的完整链路，并记录至少一次真实问题及其定位过程。任务 A 还要求从 inference optimization 或 ML Systems replication 中完成一个小实验。
- **任务 B：研究 training dynamics**。需要提出一个尚不知道答案的问题，先写出假设，再设计尽量一次只改变一个变量的 controlled experiments，分析失败和与假设冲突的结果，并形成技术报告。

本项目当前完成的实验是：

- 任务 A：比较普通自回归生成和 KV cache 推理，属于inference optimization；
- 任务 B：比较训练过程中上下文长度保持不变、从 128 增长到 512、从 512 截短到 128 等路径对模型训练动态的影响。

## 当前正式方案

当前正式方案只使用英文 TinyStories，并使用单独训练的英文 SentencePiece tokenizer。模型为 8 层、512 hidden、8 heads、最大上下文 512 token，约 3341 万参数，使用 AdamW、FP16 mixed precision、gradient accumulation 和 cosine learning-rate decay。

当前基础训练结果：

- 训练 29,297 steps，累计处理 240,001,024 tokens；
- 最终 checkpoint：`checkpoints/english_single/base/final_step_00029297.pt`；
- 最近记录的英文验证 loss：1.5810；
- 最近记录的 perplexity：4.8599。

Task A 和 Task B 均已使用英文单语方案重新完成。旧的中英混合方案仍保留在项目中，作为迁移记录、问题复盘和历史结果，不能与当前正式实验混用。

## 文档入口

项目中的详细说明分为四个根目录 TXT 文件，README 仅说明总体信息：

- [模型方案.txt](模型方案.txt)：模型架构、数据、tokenizer、训练超参数、Task A/Task B 设计和历史边界；
- [操作说明.txt](操作说明.txt)：如何开始、继续和停止训练，查看日志与 TensorBoard，生成文本，评估模型以及修改参数；
- [项目结构.txt](项目结构.txt)：项目目录、当前正式文件、历史保留文件、日志、checkpoint 和实验结果的详细位置。
- [迁移潜在问题.txt](迁移潜在问题.txt)：独立交付包已验证的可运行环节、接收方环境要求和仍需核对的迁移风险。

其他重要材料：

- [模型逐层讲解](notes/model-walkthrough.md)：从 token 到 optimizer 的代码链路；
- [Task A 报告](experiments/english_single/task_a/kv-cache-report.md)：KV cache 方法、测量结果和限制；
- [Task B 设计](notes/task-b-english-single-plan.md)：研究问题、假设和 controlled experiment 设计；
- [Task B 报告](experiments/english_single/task_b/task-b-report.md)：四条训练路径的完整结果与分析；
- [总技术报告](technical-report.md)：按照考核要求整理的总实验报告；
- [真实问题记录](notes/debug-log.md)：训练和实验中遇到的问题、定位过程和修复记录。

## 最小运行入口

所有命令默认从项目根目录运行。首次在另一台电脑解压后，先执行环境准备脚本；它会检查或重建因路径变化而失效的 `.venv`，然后安装依赖：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

环境准备完成后，当前英文基础训练的启动脚本是：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_english_single.ps1
```

测试当前英文 checkpoint 的生成示例：

```powershell
.\.venv\Scripts\python.exe -m src.sample `
  --config configs/english_single/base.yaml `
  --checkpoint checkpoints/english_single/base/final_step_00029297.pt `
  --prompt "Once upon a time" `
  --max-new-tokens 120 `
  --seed 1337
```

基础训练的 TensorBoard 日志位于：

```text
experiments/english_single/base/tensorboard/
```

Task B 的四条路径日志位于：

```text
experiments/english_single/task_b/tensorboard/
```

启动命令、参数含义、停止和恢复方法见 [操作说明.txt](操作说明.txt)。

## 代码和结果

- `src/`：模型、tokenizer、数据集、训练、评估、文本生成和实验脚本；
- `configs/english_single/`：当前英文单语方案的配置；
- `data/english_single/`：英文训练/验证文本、token 文件和数据清单；
- `tokenizer/english_single/`：当前正式 tokenizer；
- `checkpoints/english_single/`：基础训练和 Task B 的模型状态；
- `experiments/english_single/`：TensorBoard 日志、指标、对比表、报告和实验状态。

完整层级和文件作用见 [项目结构.txt](项目结构.txt)。

## 历史方案边界

项目早期曾使用英文 TinyStories 与中文 Wikipedia 混合语料。由于两种语料在语言、文体、词汇复杂度和 token 化效率上差异较大，正式方案后来改为英文 TinyStories 单语训练。旧数据、tokenizer、checkpoint、日志、配置和失败记录均保留在原位置，作为历史资料；当前英文方案使用独立的 tokenizer、数据文件、配置、checkpoint、日志和实验目录，并从随机初始化开始。

迁移原因和新旧文件边界见 [英文单语迁移记录](notes/english-single-migration.md)。

## `Doc文档/` 目录

`Doc文档/` 是个人工作目录，用来保存考核原文、讨论记录、草稿和其他文档材料。完整项目中有五份文档；独立交付包中另有 `总实验报告.docx`，共六份。它们不属于模型代码、训练数据、tokenizer 或实验结果，也不会作为训练语料使用。

## GitHub库链接

本项目存储在一个GitHub库，链接为(https://github.com/GuFurunyu/N1AISchoolEntryTestRepo)（如果你是在GitHub上看到的那么你正在这个网址）。可在该链接查看项目commit记录等信息。需要注意由于存储空间限制，训练和记录数据以及其它许多文件没有上传到GitHub，因此仅从此处下载是不完整的
