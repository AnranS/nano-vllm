from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """连续批处理调度器：决定每轮算哪些序列、做 prefill 还是 decode。"""

    def __init__(self, config: Config):
        """根据配置初始化队列上限、EOS、以及 KV 块管理器。"""
        self.max_num_seqs = config.max_num_seqs                    # 单轮最多调度多少条序列
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单轮最多处理多少 token（主要限制 prefill）
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()   # 等待 prefill / 被抢占后重新排队
        self.running: deque[Sequence] = deque()   # 已完成 prefill、正在 decode

    def is_finished(self):
        """整批是否结束：waiting / running 都空才返回 True。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """接收新请求，放入 waiting 队列等待 prefill。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选出本轮要跑的序列。

        优先从 waiting 凑 prefill 批次；凑不到再从 running 做 decode。
        返回 (seqs, is_prefill)。Prefill 与 Decode 不混在同一轮。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---------- Prefill：尽量从 waiting 里凑一批 ----------
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]  # 只窥视队首，不一定立刻 pop
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 首次调度：检查能否分配 KV 块；顺带返回可命中的前缀缓存块数
                # can_allocate == -1 表示空闲块不够，本轮不再塞新序列
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                # 真正需要算的 token = 总长 - 前缀命中部分
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 分块 prefill 的后续轮：块表已有，只算还没 cached 的那一段
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 本轮 token 预算不够装下整段，且已经有别的序列入批了 → 不再塞这条
            # 分块 prefill 只允许发生在本轮第一条序列上
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                # 正式占用 KV 块（含前缀共享的那些）
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本轮实际调度的 token 数；预算不够时截断 → 分块 prefill
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 整段 prompt 都已覆盖（含前缀缓存 + 本轮），才从 waiting 挪到 running
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # 只要凑到了 prefill 批次，本轮就只做 prefill，不做 decode
        if scheduled_seqs:
            return scheduled_seqs, True

        # ---------- Decode：从 running 里捞序列，每条只生成 1 个 token ----------
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 下一 token 可能跨入新块；空闲块不够就抢占别人（从 running 尾部踢）
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())  # 踢最晚进入的，给当前序列腾块
                else:
                    # 只剩自己还是不够 → 把自己也抢占回 waiting，本轮跳过
                    self.preempt(seq)
                    break
            else:
                # while-else：循环正常结束（没 break）才走到这里
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)  # 若刚好跨块边界，追加一块
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本轮调度的序列按原顺序放回 running 队头，供下一轮继续 decode
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占一条序列：释放其 KV 块，打回 waiting 队头等待重新 prefill。"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """模型跑完后的收尾：更新缓存、追加 token，并在结束时释放资源。"""
        for seq, token_id in zip(seqs, token_ids):
            # 本轮写满的块登记哈希，供后续请求做前缀命中
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 分块 prefill 尚未吃完 prompt：只更新缓存进度，不追加生成 token
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 碰到 EOS（且未 ignore_eos）或达到 max_tokens → 结束并释放块
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
