"""分页 Attention，去掉 nn.Module 包装。

Triton 的 store_kvcache_kernel 本来就不是 nn.Module，直接复用原版，避免重复编译
出两份一模一样的 kernel。这里替换掉的只有 Attention 这个 nn.Module 外壳。
"""
import torch

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.attention import store_kvcache
from nanovllm.utils.context import get_context


class AttentionState:
    """一层 Attention 的静态配置加它那份 KV 池视图。"""
    __slots__ = ("num_heads", "head_dim", "scale", "num_kv_heads", "k_cache", "v_cache")

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])


def attention(s: AttentionState, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    context = get_context()
    k_cache, v_cache = s.k_cache, s.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                   max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                   softmax_scale=s.scale, causal=True, block_table=context.block_tables)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                                    softmax_scale=s.scale, causal=True)
    return o
