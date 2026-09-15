from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对半切开做二维旋转（GPT-NeoX 风格）：把第 i 维和第 i+d/2 维配成一对。"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE：启动时把所有位置的 cos/sin 预先算好，运行时只做一次查表。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size    # 这里不支持只旋转一部分维度
        # 频率随维度指数衰减：低维转得快（管近距离），高维转得慢（管远距离）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)    # [位置, 频率]
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼成一张表方便一次索引；unsqueeze(1) 留出头维，后面按头广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # persistent=False：这是可重算的常量表，不写进 state_dict
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions 是绝对位置，所以分块 prefill、前缀命中后接着算都能拿到正确角度
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """所有层共用同一个 RoPE 实例：参数一致时缓存表只需要存一份。"""
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
