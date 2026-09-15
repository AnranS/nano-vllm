"""并行线性层，去掉 nn.Module 包装。

原版有 5 个类，但前向只有两种：列并行系（含普通 Linear、合并 QKV、合并
gate_up）直接做矩阵乘；行并行要在结果上 all-reduce。这里保留这两种，切分逻辑
不用重写——权重是从已经加载好的 nn.Module 模型里直接取过来的。
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


class LinearWeight:
    __slots__ = ("weight", "bias", "tp_rank", "tp_size")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, tp_rank: int = 0, tp_size: int = 1):
        self.weight = weight
        self.bias = bias
        self.tp_rank = tp_rank
        self.tp_size = tp_size


def linear(w: LinearWeight, x: torch.Tensor) -> torch.Tensor:
    """列并行 / 复制式线性层：各卡算各自那段输出，不需要通信。"""
    return F.linear(x, w.weight, w.bias)


def row_linear(w: LinearWeight, x: torch.Tensor) -> torch.Tensor:
    """行并行：各卡算部分和，再 all-reduce。bias 只在 rank0 加，避免被累加多次。"""
    y = F.linear(x, w.weight, w.bias if w.tp_rank == 0 else None)
    if w.tp_size > 1:
        dist.all_reduce(y)
    return y
