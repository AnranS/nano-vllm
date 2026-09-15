import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表按卡切分的嵌入层：每张卡只存词表的一段。

    单卡时就是普通 embedding；多卡时每张卡查自己那段，查不到的置零，
    再用 all_reduce 把各卡结果加起来拼出完整向量。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本卡负责的词表区间 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 不属于本卡的 token 先记下来，再统一映射到下标 0（避免越界索引）
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y    # 把刚才那些占位查表结果清零
            dist.all_reduce(y)           # 每个 token 只有一张卡贡献非零值，相加即还原
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """输出头。权重形状与嵌入层相同（可绑权重），但前向是矩阵乘而非查表。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # prefill 只有每条序列最后一个位置需要预测下一个 token。
            # 先按 cu_seqlens_q 的边界把这些位置挑出来，再做投影——
            # 否则要为整段 prompt 算一遍 [tokens, vocab_size] 的大矩阵，非常浪费。
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 词表维被切开了，每张卡只算出一段 logits；gather 到 rank0 拼完整
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
