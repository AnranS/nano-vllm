import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """门控 MLP 的激活：输入是 gate 和 up 拼在一起的一张大张量，对半切后相乘。

    合并成一次矩阵乘再切开，比分别算 gate_proj / up_proj 少一次 kernel 启动。
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
