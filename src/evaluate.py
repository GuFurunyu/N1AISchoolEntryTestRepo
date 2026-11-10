"""读取已保存的模型，在中英文验证语料上计算 loss 和困惑度。

这里不更新参数。与训练过程中的周期验证一样，结果反映模型预测下一个
token 的能力；困惑度是平均交叉熵取指数，不是文本质量的完整评价。
"""

import argparse
import math

import torch

from .config import load_config
from .dataset import make_eval_loader
from .model import TransformerLM


@torch.no_grad()
def main():
    """从命令行读取检查点与批次数，分别评估两种语言。"""
    parser = argparse.ArgumentParser()
    # --checkpoint 是必须提供的权重文件；--batches 决定每种语言取多少批。
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    model_config, data_config, train_config = load_config(args.config)
    # 根据当前机器选择设备；即使配置要求 CUDA，不可用时仍能在 CPU 运行。
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    # 文件中还有优化器等信息；评估只需要 model 参数。
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    # eval 关闭 dropout，使同一输入的预测不受训练期随机失活影响。
    model.eval()
    losses = {}
    # val.bin 的前段是英语、后段是中文；英语末尾留出 block_size 以免
    # 一个验证窗口跨越语言边界。start/end 是 token 位置，不是字节位置。
    spans = {
        "english": (0, data_config.val_english_tokens - model_config.block_size),
        "chinese": (data_config.val_english_tokens, None),
    }
    for language, (start, end) in spans.items():
        # 固定窗口长度和批大小，与训练时的验证口径保持一致。
        loader = make_eval_loader(data_config.val_bin, model_config.block_size, train_config.micro_batch_size, args.batches, train_config.num_workers, start, end)
        iterator = iter(loader)
        total = 0.0
        for _ in range(args.batches):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                # 验证迭代器耗尽时重建，以取满 --batches 批。
                iterator = iter(loader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                # 只关心 targets 对应的平均交叉熵，不需要 logits。
                _, loss = model(inputs, targets)
            total += float(loss)
        losses[language] = total / args.batches
        print(f"{language}_loss={losses[language]:.5f}")
    weighted = 0.65 * losses["english"] + 0.35 * losses["chinese"]
    # 这里按人为固定权重合并两种语言；不是验证文件自然比例。
    print(f"validation_loss_65en_35zh={weighted:.5f}")
    # 防止极端大的 loss 导致 exp 溢出；一般训练结果不会触及 20。
    print(f"perplexity_65en_35zh={math.exp(min(20.0, weighted)):.5f}")


if __name__ == "__main__":
    # python -m src.evaluate 才进入此入口；import 不触发评估。
    main()
