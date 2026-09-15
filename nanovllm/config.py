import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """引擎的全部旋钮。

    注意后三个字段不是用户填的，而是启动过程中被回填的：hf_config 在这里读，
    eos 由 LLMEngine 从 tokenizer 取，num_kvcache_blocks 由 ModelRunner 按
    实际剩余显存算出来。所以这个对象同时充当了跨组件的共享白板。
    """
    model: str
    max_num_batched_tokens: int = 16384    # 单轮 token 预算，主要约束 prefill
    max_num_seqs: int = 512                # 单轮最多调度多少条序列
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9    # 允许占用的显存比例，剩下的留给 KV 池之外的开销
    tensor_parallel_size: int = 1
    enforce_eager: bool = False            # True 则跳过 CUDA Graph，调试时更容易跟踪
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)    # 只接受本地目录，不会去下载权重
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 用户设的上限不能超过模型自身的位置编码范围
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
