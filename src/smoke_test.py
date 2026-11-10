"""用合成数据做快速自检，检查前向、反向和因果注意力。

这里不读取训练语料，也不评价最终语言能力。它适合在环境搭好后
先检查模型的基本计算能否运行。
"""

import argparse

import torch

from .config import load_config, ModelConfig
from .model import TransformerLM


def main():
    """构造小批随机输入，尝试更新参数，再检验未来 token 是否泄漏。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    model_config, _, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    print(f"device={device}")
    print(f"parameters={model.num_parameters():,}")

    # 缩短序列以减少自检成本；形状是 [batch, seq_len]。
    batch, seq_len = 2, min(64, model_config.block_size)
    # inputs 是 token ID，targets 是随机目标；此处只测计算链条是否通畅。
    inputs = torch.randint(0, model_config.vocab_size, (batch, seq_len), device=device)
    targets = torch.randint(0, model_config.vocab_size, (batch, seq_len), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    initial_loss = None
    final_loss = None
    for step in range(20):
        # 在固定的同一批随机样本上重复训练，检查 loss 和梯度计算。
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets)
        if initial_loss is None:
            initial_loss = float(loss)
        loss.backward()
        # backward 计算每个可训练参数的梯度，step 用这些梯度更新参数。
        optimizer.step()
        final_loss = float(loss)
    assert logits.shape == (batch, seq_len, model_config.vocab_size)
    # loss 必须是有限数值；本脚本并未断言 final_loss 一定小于 initial_loss。
    assert torch.isfinite(loss)
    print(f"initial_loss={initial_loss:.5f}")
    print(f"final_loss={final_loss:.5f}")
    model.eval()
    # 把序列后半段替换成新 token；因果掩码应保证前半段输出不变。
    changed_future = inputs.clone()
    changed_future[:, seq_len // 2 :] = torch.randint(
        0, model_config.vocab_size, changed_future[:, seq_len // 2 :].shape, device=device
    )
    with torch.no_grad():
        # 关闭梯度构图，仅比较两种输入的预测。
        original_logits, _ = model(inputs)
        changed_logits, _ = model(changed_future)
    assert torch.allclose(
        # atol/rtol 允许浮点计算造成的微小差别。
        original_logits[:, : seq_len // 2], changed_logits[:, : seq_len // 2],
        atol=1e-5, rtol=1e-5,
    ), "Future tokens changed earlier logits; check the causal mask"
    print("causal_mask_check=PASS")
    print("forward_backward_check=PASS")


if __name__ == "__main__":
    # 直接运行时才执行自检。
    main()
