import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm，并可选地把残差相加融合进来。

    Transformer 主干里每层都是"加残差 → 归一化"。把这两步放进同一个编译单元，
    中间结果不用落回显存；同时返回加完残差的值，供下一层继续累加。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """不带残差的版本，用于每层的第一个 norm。"""
        orig_dtype = x.dtype
        # 平方和在 fp16/bf16 下容易溢出或精度不足，统一升到 fp32 再算
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """先加残差再归一化，返回 (归一化结果, 新的残差)。"""
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)    # 残差流保存的是"加完、未归一化"的值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
