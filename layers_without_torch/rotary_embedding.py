"""RoPE，去掉 nn.Module 包装。

原版把 cos/sin 表注册成 buffer，于是每次 forward 访问 self.cos_sin_cache 都要
经过 nn.Module.__getattr__ 去 _buffers 里捞。这里它就是个普通属性。
"""
import torch


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RopeCache:
    """预计算好的 cos/sin 表。所有层共用同一个实例，和原版的 lru_cache 等效。"""
    __slots__ = ("cos_sin_cache",)

    def __init__(self, cos_sin_cache: torch.Tensor):
        self.cos_sin_cache = cos_sin_cache


@torch.compile
def rope_forward(cos_sin_cache: torch.Tensor, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return query, key


def apply_rope(rope: RopeCache, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
    return rope_forward(rope.cos_sin_cache, positions, query, key)
