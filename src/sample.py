"""从检查点逐 token 生成文本，便于直观看到模型学到的语言模式。"""

import argparse

import torch

from .config import load_config
from .model import TransformerLM
from .tokenizer import load_tokenizer


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature, top_k, use_kv_cache=False):
    """自回归采样：反复预测下一个 token，并把它接到输入末尾。

input_ids 的形状为 [batch, 已有 token 数]；本脚本主入口只传入 batch=1。
返回值包含原提示词和所有新生成的 token。默认路径每次重算上下文；
use_kv_cache=True 时各层保存旧 K/V，只计算新增 token。
"""
    model.eval()
    past_key_values = None
    for _ in range(max_new_tokens):
        if use_kv_cache:
            if past_key_values is None or past_key_values[0][0].size(-2) >= model.config.block_size:
                # 首次处理完整提示词。超过窗口上限时重新计算最近的 block_size
                # 个 token，以保持与无缓存路径相同的截断和 RoPE 位置编号。
                context = input_ids[:, -model.config.block_size:]
                logits, past_key_values = model.forward_cached(context)
            else:
                # 上一轮已缓存提示词/历史 token，本轮只需计算刚生成的一个。
                logits, past_key_values = model.forward_cached(input_ids[:, -1:], past_key_values)
        else:
            # 原路径每轮都重新处理最近的 block_size 个 token。
            context = input_ids[:, -model.config.block_size:]
            logits, _ = model(context)
        # logits[:, -1, :] 是最后一个位置对整个词表的下一个 token 评分。
        # 温度越低，softmax 后概率越集中；下限避免除以零。
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            # 只让评分最高的 k 个 token 参与抽样，其余设为负无穷。
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probabilities = torch.softmax(logits, dim=-1)
        # 按概率随机选一个 token，而非总选得分最高者。
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def main():
    """读取命令行选项、恢复模型、编码提示词，再解码生成结果。"""
    parser = argparse.ArgumentParser()
    # temperature 控制随机程度；top-k 限定候选 token 数。
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    # 加此开关启用 KV cache；不加则继续使用原来的全上下文重算路径。
    parser.add_argument("--kv-cache", action="store_true")
    args = parser.parse_args()
    model_config, data_config, train_config = load_config(args.config)
    device = torch.device("cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    # 固定采样随机种子，以便相同环境和参数下重复生成。
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = load_tokenizer(data_config.tokenizer_model)
    model = TransformerLM(model_config).to(device)
    # 完整检查点还含有优化器状态；先读入 CPU，只把模型参数复制到目标设备。
    # 否则无用的优化器状态也会占用 GPU 显存，污染推理显存测量。
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    encoded = tokenizer.encode(args.prompt, out_type=int)
    # 外层方括号构成 batch 维；dtype=long 是 embedding 索引的要求。
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    output = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_k, args.kv_cache)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    # 只有直接运行模块时才生成文本。
    main()
