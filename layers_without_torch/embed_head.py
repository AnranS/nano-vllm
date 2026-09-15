"""词表切分的嵌入层与输出头，去掉 nn.Module 包装。"""
import torch
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabWeight:
    __slots__ = ("weight", "tp_rank", "tp_size", "vocab_start_idx", "vocab_end_idx")

    def __init__(self, weight: torch.Tensor, tp_rank: int, tp_size: int, vocab_start_idx: int, vocab_end_idx: int):
        self.weight = weight
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.vocab_start_idx = vocab_start_idx
        self.vocab_end_idx = vocab_end_idx


def embedding(w: VocabWeight, x: torch.Tensor) -> torch.Tensor:
    if w.tp_size > 1:
        mask = (x >= w.vocab_start_idx) & (x < w.vocab_end_idx)
        x = mask * (x - w.vocab_start_idx)
    y = F.embedding(x, w.weight)
    if w.tp_size > 1:
        y = mask.unsqueeze(1) * y
        dist.all_reduce(y)
    return y


def lm_head(w: VocabWeight, x: torch.Tensor) -> torch.Tensor:
    context = get_context()
    if context.is_prefill:
        # prefill 只有每条序列最后一个位置要预测，先挑出来再投影
        last_indices = context.cu_seqlens_q[1:] - 1
        x = x[last_indices].contiguous()
    logits = F.linear(x, w.weight)
    if w.tp_size > 1:
        all_logits = [torch.empty_like(logits) for _ in range(w.tp_size)] if w.tp_rank == 0 else None
        dist.gather(logits, all_logits, 0)
        logits = torch.cat(all_logits, -1) if w.tp_rank == 0 else None
    return logits
