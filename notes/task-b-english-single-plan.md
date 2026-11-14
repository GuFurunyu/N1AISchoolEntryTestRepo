# 英文单语 Task B：训练上下文在中点增长或截短

> 记录时间说明：本文件是在英文实验完成后整理的设计说明，不是英文实验的预注册。
> 四路径问题、H1/H2/H3 和主要控制方法已见更早的旧中文方案计划
> `notes/task-b-plan.md`；英文重建沿用了这一研究设计。下文“事前假设”
> 指原计划的假设来源，不表示本文件在英文训练开始前写成。

## 研究问题

在相同英文 TinyStories 数据、相同模型初始化和相同优化器预算下，如果训练过程前半段使用 128 token 上下文，后半段改为 512 token，模型获得长历史信息的能力如何变化？反过来，如果前半段使用 512，后半段改为 128，长上下文能力会不会保留、减弱或被短上下文能力替代？

这里研究的是训练时每个目标 token 实际能看到的历史长度，而不是生成时把 prompt 截短。四条路径都真正更新参数，因此可以观察训练 dynamics。

## 事前假设

1. `parent128` 在 512 token 验证历史下的表现会明显差于 `parent512`，因为 128 token 训练窗口在每个窗口边界重置可见历史。
2. `128-512` 会在切换后逐渐降低 `L512`，并把 `G=L128-L512` 从负值推向正值；`128-128` 作为对照不会获得同样的长上下文收益。
3. `512-128` 仍可能改善 `L128`，但 `G` 会比 `512-512` 小，说明短窗口训练可能削弱而不是立即消除已经形成的长历史利用能力。

## 控制设计

- 数据：只读 `data/english_single/processed/train.bin` 和 `val.bin`，由 `data/english_single/manifest.json` 中的 SHA-256 固定身份。
- tokenizer：只使用 `tokenizer/english_single/spm.model`；旧中英 tokenizer 不在新配置中出现。
- 模型：8 层、512 hidden、8 heads、512 最大 context、约 33.4M 参数；四条路径从同一个随机初始化状态分叉。
- 每个优化步骤处理 8192 个目标 token；每个父阶段 9766 步、80,003,072 token；每条完整路径 160,006,144 token。
- 父阶段：`parent128` 和 `parent512` 分别使用 128 或 512 token，训练到同一 token 中点。
- 子阶段：`128-128`、`128-512`、`512-512`、`512-128` 从对应父 checkpoint 复制模型、AdamW、GradScaler、学习率进度、数据 RNG 和 PyTorch RNG。
- 数据抽样、验证窗口、验证目标尾部、温度、top-k 和生成 seed 固定；改变的变量只有第二阶段的训练 context。

## 测量

固定 64 个英文验证窗口。对每个窗口分别提供 128、256、512 token 历史，只计算最后相同的 128 个目标位置交叉熵：`L128`、`L256`、`L512`。主要差值为 `G=L128-L512`；同时报告原始 loss，防止只看差值而误判能力变化。生成样例只作为补充，不能替代 loss。

## 可复核文件

- 设计：本文件。
- 数据和配置快照：`experiments/english_single/task_b/manifest.json`、各路径 `effective_config.json`。
- 长表：`experiments/english_single/task_b/comparison.csv`。
- 宽表：`experiments/english_single/task_b/comparison_wide.csv`。
- 逐路径日志：各路径 `metrics.jsonl`、`samples.json` 和 TensorBoard event 文件。
- 完整状态：`checkpoints/english_single/task_b/`。
