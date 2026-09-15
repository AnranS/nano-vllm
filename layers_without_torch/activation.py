"""SiluAndMul，去掉 nn.Module 包装。

原版是个无状态的 nn.Module——没有任何参数，存在的唯一理由就是能被 nn.Sequential
式地组装。拆成自由函数后，调用少一层 Module.__call__ 的分派。
"""
import torch
import torch.nn.functional as F


@torch.compile
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
