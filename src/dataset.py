"""从二进制 token 文件构造“根据前文预测下一个 token”的训练样本。

训练文件是连续的 uint16 ID。memory map 让操作系统按需读取窗口，
不要求一次把一亿多个 ID 全部装入 Python 内存。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset


class MemmapTokenDataset(Dataset):
    """给定起始位置 i，返回错开一个 token 的输入和目标序列。"""

    def __init__(self, path: str, block_size: int):
        # Path 提供跨平台的文件路径接口；这里要求编码文件已经存在。
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token file does not exist: {self.path}")
        # mode="r" 只读；dtype 必须与 prepare_data.py 写入时的 uint16 一致。
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        # 长度为 T 的输入需要 T+1 个 ID，才有最后一个位置的正确答案。
        if len(self.tokens) <= block_size:
            raise ValueError(f"Token file {self.path} is shorter than block size")

    def __len__(self):
        """合法起始位置的数量，而非文件中的 token 总数。"""
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        """读取 T+1 个 ID，构造两条长度为 T 的错位序列。

        例：window=[A,B,C,D] 时，输入=[A,B,C]，目标=[B,C,D]。
        每个位置都学习预测它后面的 token。
        """
        # 复制小窗口，并转成 PyTorch embedding/cross_entropy 使用的 int64。
        # 这样生成的张量不再共享只读 memmap 的底层内存。
        window = np.array(self.tokens[index : index + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


class BilingualSampler(Sampler[int]):
    """先选语言，再在该语言的 token 区间内抽随机窗口起点。

编码文件的英文部分在前、中文部分在后。这里返回的是“起始索引”，
不是 token ID；DataLoader 随后用索引调用上面的 __getitem__。
"""

    def __init__(self, dataset: MemmapTokenDataset, english_tokens: int, english_probability: float):
        # 英文区和中文区都至少要容得下一个完整的训练窗口。
        if not (dataset.block_size < english_tokens < len(dataset.tokens) - dataset.block_size):
            raise ValueError("English token boundary is outside the usable dataset")
        if not 0.0 <= english_probability <= 1.0:
            raise ValueError("english_probability must be between 0 and 1")
        self.dataset = dataset
        # 边界是英文 ID 总数，也就是中文区第一个 ID 的位置。
        self.english_tokens = english_tokens
        self.english_probability = english_probability

    def __len__(self):
        """一个采样器周期可产生的起始索引数。"""
        return len(self.dataset)

    def __iter__(self):
        """按 1024 个索引一组生成，避免一次分配巨大的随机排列。"""
        remaining = len(self)
        while remaining > 0:
            chunk = min(remaining, 1024)
            # True 表示这一个窗口抽英文；随机抽样允许窗口重复出现。
            choose_english = torch.rand(chunk) < self.english_probability
            # 英文窗口的最后一个目标 token 也必须留在英文区内。
            en_indices = torch.randint(0, self.english_tokens - self.dataset.block_size, (chunk,))
            # 中文窗口从边界开始，到数据集允许的最后一个起点结束。
            zh_indices = torch.randint(self.english_tokens, len(self.dataset), (chunk,))
            # 按语言选择每个起点，转换为 Python 整数并逐个交给 DataLoader。
            indices = torch.where(choose_english, en_indices, zh_indices)
            yield from indices.tolist()
            remaining -= chunk


def make_loader(path: str, block_size: int, batch_size: int, shuffle: bool, num_workers: int,
                english_tokens: int = 0, english_probability: float = 0.5):
    """构造训练 DataLoader，自动把多个 [T] 样本堆成 [B,T]。"""
    dataset = MemmapTokenDataset(path, block_size)
    # 训练调用方会给 shuffle=True 和英文边界，于是采用分语言随机采样器。
    # 未提供边界时按文件原有顺序读取；这里没有调用 DataLoader 内置 shuffle。
    sampler = BilingualSampler(dataset, english_tokens, english_probability) if shuffle and english_tokens else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        # GPU 可用时锁定 CPU 内存页，有助于 CPU->GPU 复制。
        pin_memory=torch.cuda.is_available(),
        # 不输出不满 batch_size 的最后一批，使训练时 batch 形状固定。
        drop_last=True,
    )


def make_eval_loader(path: str, block_size: int, batch_size: int, batches: int, num_workers: int,
                     start: int = 0, end: int = None):
    """在指定 token 区间选固定、均匀分散的窗口用于验证。

start/end 是起始位置的范围；英文和中文分别调用此函数，才能分别
计算两个 loss。固定索引让不同 checkpoint 的验证结果可比较。
"""
    dataset = MemmapTokenDataset(path, block_size)
    end = len(dataset) if end is None else end
    if not 0 <= start < end <= len(dataset):
        raise ValueError("Invalid validation token span")
    # 最多取 batches 个完整 batch；区间较短时不超过可用起点数。
    sample_count = min(end - start, batch_size * batches)
    # linspace 在头尾之间均匀选位置，不只评估文件开头。
    indices = np.linspace(start, end - 1, sample_count, dtype=np.int64).tolist()
    return DataLoader(
        # Subset 把原始数据集限制为这些固定位置，顺序保持不变。
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
