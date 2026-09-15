from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """请求的三种状态：排队等 prefill、正在 decode、已结束。"""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """一条请求的完整账本：token、KV 块表、采样参数、进度计数。

    它是贯穿整个引擎的唯一状态载体：scheduler 读它决定本轮算谁，
    model_runner 读它拼 GPU 输入，postprocess 把新 token 写回来。

    三个长度计数最容易混：
      num_tokens           总长度，包含刚采样出、但还没算过 KV 的那个 token
      num_cached_tokens    已经算好 KV 的 token 数（含前缀命中的部分）
      num_scheduled_tokens 本轮准备送进模型的 token 数，postprocess 后清零
    """
    block_size = 256    # 类变量：所有序列共用，LLMEngine 启动时按 config 覆盖
    counter = count()   # 全局自增计数器，给每条请求发 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)            # 拷贝一份，避免和调用方共享同一个 list
        self.last_token = token_ids[-1]             # 单独存一份：decode 每轮只需要它
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)     # 之后不再变，用来切分 prompt / completion
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []                       # 逻辑块 i → KV 池里的物理块号
        # 采样参数摊平成普通字段，省得每轮再穿一层对象
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """容纳当前所有 token 需要几个块（向上取整，末块通常没填满）。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """末块里实际占了几个位置，取值范围 1..block_size。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """取第 i 个逻辑块的 token 切片，用于算块哈希、做前缀比对。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """pickle 时只带上调度必需的字段（TP>1 时 rank0 要把序列发给其他 worker）。

        prefill 阶段 worker 需要完整 token_ids 才能算 KV；decode 阶段每轮只送
        1 个 token，传 last_token 就够。这样 decode 的进程间消息是常数大小，
        不会随着对话变长而线性膨胀。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """worker 侧还原。seq_id / status / 采样参数不在其中——只有 rank0 用得到。"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode：没有历史 token_ids，只还原最后一个 token
            self.token_ids = []
            self.last_token = last_state
