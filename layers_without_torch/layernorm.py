"""RMSNorm，去掉 nn.Module 包装。

计算过程和 nanovllm/layers/layernorm.py 逐行一致，区别只在于：
权重是裸 Tensor 而不是 nn.Parameter，前向是自由函数而不是方法。
"""
import torch


class RMSNormWeight:
    """只存权重和 eps 的数据容器。

    用 __slots__ 而不是普通 class：属性走 slot 描述符，比 __dict__ 查找快，
    更关键的是完全绕开 nn.Module.__getattr__ 里那串 _parameters / _buffers /
    _modules 的字典链——原版每次访问 self.weight 都要走一遍。
    """
    __slots__ = ("weight", "eps")

    def __init__(self, weight: torch.Tensor, eps: float):
        self.weight = weight
        self.eps = eps


@torch.compile
def rms_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x


@torch.compile
def add_rms_forward(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual


def rms_norm(w: RMSNormWeight, x: torch.Tensor, residual: torch.Tensor | None = None):
    """对应原版 RMSNorm.forward 的分派。"""
    if residual is None:
        return rms_forward(x, w.weight, w.eps)
    return add_rms_forward(x, residual, w.weight, w.eps)
