"""一个从零训练的 decoder-only Transformer 语言模型。

训练目标：读入前面的 token ID，并在每个位置预测下一个 token。
本文件显式写出 Q/K/V、遮罩、softmax 和加权求和，方便逐行理解 attention。
核心形状约定：B=批量中的序列数，T=每条序列的 token 数，C=向量维度。
默认配置是 T<=512、C=512、8 个 head、每个 head 64 维。
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RotaryEmbedding(nn.Module):
    """预计算 RoPE 的正弦/余弦，并按位置旋转每个 head 的 Q、K。

位置不会直接加到 token embedding 上。RoPE 在 attention 比较 Q 和 K
之前改变它们的方向，使点积能够反映 token 的相对位置关系。
"""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        # 最后一个维度必须能分成相同大小的两半，形成旋转用的二维配对。
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        # 每对维度使用不同旋转频率；维度越靠后，频率通常越低。
        # 默认 head_dim=64 时，arange(0,64,2) 有 32 个元素。
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # 位置编号是 0 到 max_seq_len-1，不是 token ID。
        positions = torch.arange(max_seq_len).float()
        # 外积得到 [位置数, 维度对数]，每个元素是该位置在该维度对的角度。
        frequencies = torch.outer(positions, inv_freq)
        # 复制频率，使余弦/正弦能逐元素对应完整的 head_dim 维向量。
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        # [1,1,T,head_dim] 的前两个单例维可广播到 batch 与 head。
        # register_buffer 会随模型移动到 GPU，但它不是可训练参数；
        # persistent=False 表示不把可重算的表存进 checkpoint。
        self.register_buffer("cos", embedding.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", embedding.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """把末维两半 [a,b] 变成 [-b,a]，相当于二维旋转所需的 90° 项。"""
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """对形状 [B,head,T,head_dim] 的 Q、K 应用位置旋转。

        普通前向从位置 0 开始；缓存解码时，新 token 的位置是已缓存的长度。
        """
        seq_len = q.size(-2)
        # 只截取当前新 token 对应的位置；普通训练的 start_pos 默认为 0。
        cos = self.cos[:, :, start_pos : start_pos + seq_len].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, start_pos : start_pos + seq_len].to(dtype=q.dtype, device=q.device)
        # 这是 x*cos(theta)+rotate_half(x)*sin(theta) 的逐维形式。
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：每个位置只读取自己和更早的位置。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # 512 维分给 8 个 head，每个 head 得到 64 维。
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # 一个 512->1536 投影一次产出 Q、K、V，随后拆成三个 512 维张量。
        # 它等价于三个独立的 512->512 线性层，写在一起便于计算。
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 8 个 head 拼接后的结果仍为 512 维，再投影回残差流。
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 训练时 attention 权重与输出各自使用 dropout；eval() 时关闭。
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size, config.rope_base)
        # 上三角 i<j 表示 query 位置 i 试图读取未来 key 位置 j。
        # 对角线不遮住，当前位置可以看到自己。
        mask = torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1)
        # mask 是固定结构，不学习，也不需要存入 checkpoint。
        self.register_buffer("causal_mask", mask[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入/输出均为 [B,T,512]，中间显式计算注意力权重。"""
        batch, seq_len, channels = x.shape
        # qkv(x) 为 [B,T,1536]，split 后三者各为 [B,T,512]。
        q, k, value = self.qkv(x).split(channels, dim=-1)
        # view 先变 [B,T,8,64]，transpose 再变 [B,8,T,64]。
        # 这样每个 head 就能独立计算 T×T 的位置关联。
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # 位置旋转只作用于 Q、K；V 保留待取出的内容表示。
        q, k = self.rope(q, k)

        # Q×K^T 的形状是 [B,8,T,T]；每个分数表示 query 对某个 key 的匹配。
        # 除以 sqrt(64) 避免维度较大时点积尺度过大，使 softmax 饱和。
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 未来位置改为 -inf；softmax 后相应概率为 0，防止训练时偷看答案。
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        # 对最后一维的所有可见 key 做归一化：每个 query 得到一组注意力权重。
        weights = F.softmax(scores, dim=-1)
        # dropout 只在训练时随机置零一部分权重，以抑制过拟合。
        weights = self.attn_dropout(weights)
        # 对 V 按注意力权重加权求和，结果形状为 [B,8,T,64]。
        attended = weights @ value
        # 交换 head 和 T，再拼接 8 个 head，恢复 [B,T,512]。
        # contiguous() 确保 transpose 后的内存布局可供 view 使用。
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.proj(attended))

    def forward_cached(
        self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """仅对新 token 算 Q/K/V，并复用本层之前保存的 K/V。

        past_key_value 中的两个张量形状均为 [B,head,旧长度,head_dim]。
        首次处理完整提示词时它是 None；之后通常一次只传入一个新 token。
        返回本次注意力输出，以及供下一次调用使用的完整 K/V。
        """
        batch, new_len, channels = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        total_len = past_len + new_len
        if total_len > self.causal_mask.size(-1):
            raise ValueError("KV cache exceeds block size; rebuild it from the latest context")

        q, k, value = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, new_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, new_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, new_len, self.n_head, self.head_dim).transpose(1, 2)
        # 只对新 Q/K 按它们的真实位置做旋转；旧 K 已在之前的调用中旋转。
        q, k = self.rope(q, k, start_pos=past_len)
        if past_key_value is not None:
            # 沿时间维拼接旧、新 K/V；旧位置不再经过投影或 RoPE。
            k = torch.cat((past_key_value[0], k), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 行只对应本次新 query 的绝对位置；列包含旧、新全部 key。
        mask = self.causal_mask[:, :, past_len:total_len, :total_len]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, new_len, channels)
        return self.resid_dropout(self.proj(attended)), (k, value)


class MLP(nn.Module):
    """逐位置前馈网络；不直接在不同 token 之间交换信息。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # 中间维度扩大 4 倍，为非线性变换提供更宽的表示空间。
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.fc2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,512] -> [B,T,2048] -> GELU -> [B,T,512]。"""
        # GELU 提供非线性；approximate="tanh" 使用较快的近似公式。
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    """一个 Pre-LN block：先 attention 残差，再 MLP 残差。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """两次残差相加都保持 [B,T,512] 的形状。"""
        # LayerNorm 在子模块之前，所以称 Pre-LN；残差保留原输入信息。
        x = x + self.attn(self.ln_1(x))
        # MLP 处理已经融合了上下文的每个位置的向量。
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_cached(
        self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """保留原 block 的 Pre-LN 与残差顺序，同时返回本层更新后的 K/V。"""
        attended, present_key_value = self.attn.forward_cached(self.ln_1(x), past_key_value)
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value


class TransformerLM(nn.Module):
    """完整语言模型：ID -> embedding -> 8 个 block -> logits -> 可选 loss。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 可训练查找表，每个 token ID 对应一个 512 维向量。
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # ModuleList 保证每一层自己的权重都注册到 PyTorch 模型中。
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # 最后一个归一化之后才把隐藏向量映射到词表分数。
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 输出层的权重与输入 embedding 指向同一参数对象，称为 weight tying。
        # 这样输入和输出共用 [16000,512] 的矩阵，减少参数量。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        # 对各线性层和 embedding 设置随机初值；训练随后更新它们。
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """给线性层/Embedding 正态分布初值，并把线性层 bias 清零。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """返回 (logits, loss)；没有 targets 时 loss 为 None。

        input_ids/targets: [B,T]；logits: [B,T,vocab_size]；loss: 标量。
        targets 应是 input_ids 向右移一位后的正确 token ID。
        """
        batch, seq_len = input_ids.shape
        # RoPE 表和因果 mask 只预生成到 block_size，输入不能更长。
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")
        # 整数 ID 先查表成为浮点向量，形状 [B,T,512]。
        x = self.token_embedding(input_ids)
        # 8 层依次处理；每一层都保留 [B,T,512] 的形状。
        for block in self.blocks:
            x = block(x)
        # 每个位置给整个 16000 词表打分，尚未转换成概率。
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            # 展平 B 和 T，得到 [B*T,vocab] 与 [B*T]；交叉熵取各位置平均。
            # cross_entropy 内部稳定地计算 log-softmax，无需手工先做 softmax。
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]:
        """推理专用前向：返回新位置的 logits 和各层更新后的 K/V。

        input_ids 只含本次新增的 token；第一次调用可传完整提示词。
        本方法本身不关闭梯度；调用方应使用 model.eval() 和
        torch.inference_mode()，避免生成时保留反向传播计算图。
        """
        if self.training:
            raise RuntimeError("forward_cached requires model.eval()")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("KV cache layer count does not match the model")
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if input_ids.size(1) == 0 or past_len + input_ids.size(1) > self.config.block_size:
            raise ValueError("Cached context must contain 1..block_size tokens")
        x = self.token_embedding(input_ids)
        present_key_values = []
        for layer, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[layer]
            x, present = block.forward_cached(x, past)
            present_key_values.append(present)
        logits = self.lm_head(self.ln_f(x))
        return logits, present_key_values

    def num_parameters(self) -> int:
        """统计可训练参数数量；共享的 embedding/输出权重只计一次。"""
        return sum(parameter.numel() for parameter in self.parameters())
