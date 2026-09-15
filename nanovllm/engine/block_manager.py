from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV 池里一个块的 CPU 侧元信息。真正的 K/V 向量在 GPU 上，这里只记账。"""

    def __init__(self, block_id):
        self.block_id = block_id    # 与 GPU 池里的物理块一一对应，创建后不变
        self.ref_count = 0          # 有几条序列正在引用；归 0 才能回收
        self.hash = -1              # 链式前缀哈希；-1 表示还没写满、不可复用
        self.token_ids = []         # 块内 token，命中哈希后还要用它做一次精确比对

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """块被重新分配出去：引用数置 1，清掉旧的哈希与内容。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """分页 KV 的块池：负责分配、前缀复用、引用计数和回收。

    池里的块只有两种归属：used_block_ids（有人引用）或 free_block_ids（空闲）。
    关键点是空闲块并没有被清空——它的 hash 和 token_ids 还留着，所以后来的
    请求仍可能命中它并原地捞回来，这就是前缀缓存能跨请求生效的原因。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()   # 前缀哈希 → 块号，前缀命中的索引
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """链式哈希：把上一块的哈希拌进来，保证"相同 token 不同前文"不会误命中。"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲队列取一块并重置。若它还挂着旧哈希索引，先把索引摘掉。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这块空闲期间一直可被前缀命中；现在要被覆盖，索引必须同步失效
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """放回空闲队列。注意不清 hash / token_ids，留着给后续请求命中。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """试算这条序列能否入场，返回可复用的前缀块数；空闲块不够则返回 -1。

        只查询不改状态，scheduler 用它决定要不要把序列放进本轮批次。
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 跳过最后一个逻辑块：末块留给本次重算，以保证能产出当前这步的 logits
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 哈希可能碰撞，命中后还要逐 token 核对；前缀必须从头连续匹配，一断就停
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 命中且正被别人用 → 直接加引用，不占空闲名额；
            # 命中但已空闲 → 还得从 free 队列里捞回来，所以仍算一个名额
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """真正建立块表：前 num_cached_blocks 个复用，其余新开。"""
        assert not seq.block_table
        h = -1
        # 复用段：哈希必须按同样顺序重算一遍，才能拿到每一块的块号
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # 空闲但内容仍有效：从 free 队列捞回来，KV 不用重算
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 未命中段：一次性把整个 prompt 剩下的块都占住（分块 prefill 切的是计算量，不是显存）
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """序列结束或被抢占：逐块减引用，减到 0 的放回空闲队列。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """下一轮 decode 是否有块可用。

        余数为 1 说明上轮采样出的 token 刚好是新块的第一个，需要额外一块；
        否则写在当前末块里，0 块就够。布尔值在这里被当成 0/1 用。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """跨到新块时补一块。判断的是已经含上轮采样结果的长度。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """本轮刚写满的块登记哈希，让后续请求能命中它们。

        在 postprocess 里、num_cached_tokens 更新之前调用，所以 [start, end)
        正好是本轮新填满的完整块；末尾没填满的块不登记（哈希留 -1）。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 从前一块的哈希接上，保持整条前缀链一致
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
