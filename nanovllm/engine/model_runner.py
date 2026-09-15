import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """一张 GPU 上的执行器：持有模型分片、KV 池和 CUDA Graph，负责跑一轮前向。

    每个 rank 一个实例。rank0 活在主进程里，由 LLMEngine 直接调用；rank1..N-1
    活在各自的子进程里，通过共享内存接收命令。scheduler 只和 rank0 打交道。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event    # rank0 持有全部 worker 的 Event；worker 只持有自己的

        # 单卡也要建进程组：各层里的 dist.get_rank() / all_reduce 才有统一入口
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        # 临时改默认 dtype/device，让模型直接在 GPU 上以权重精度构造，省掉一次搬运
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 顺序不能换：先预热量出显存峰值，才能算出 KV 池能开多大，最后才捕获图
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()    # 等 worker 都到齐，再让它们来 attach 这块共享内存
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # worker 在构造函数里就进入收命令循环，直到收到 exit 才返回
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()    # 只有创建者负责销毁
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """worker 主循环：读一条命令、执行、重复，直到 exit。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """阻塞等 rank0 的 Event，然后从共享内存取出一条命令。"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")    # 前 4 字节是负载长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """rank0 广播一条命令：写共享内存，再逐个点亮 worker 的 Event。

        这里走的是 CPU 通道，只传方法名和序列元信息（见 Sequence.__getstate__）；
        模型张量之间的通信由 NCCL 负责，不经过这块共享内存。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """统一入口：rank0 调用时顺带把同一条命令广播给其他 worker。"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """跑一次最大规模的假 prefill，把激活值的显存峰值真实地压出来。

        这些假序列没有 block_table，所以不会写 KV（此时 kv_cache 也还没分配）。
        目的只是让分配器记录峰值，供 allocate_kv_cache 反推剩余预算。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """用"剩下多少显存"反推 KV 池能开多少块，然后一次性申请。

        块数不是写死的：先按 gpu_memory_utilization 算出允许占用的上限，减去
        当前实际占用。峰值 peak 里含预热时的临时激活，那部分待会儿会被释放，
        所以用 `- peak + current` 把这段瞬时开销折回来，避免把池开得过小。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free    # 含本进程权重、分配器缓存以及卡上其他进程
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size    # TP 下每卡只存自己那几个 KV 头
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一个块的开销：K 和 V 各一份 × 每层一份 × 块内 token 数 × 每 token 的向量大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 把池按层切成视图挂到各个 Attention 上；视图共享底层显存，不额外拷贝
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把长短不一的块表补齐成矩形张量，-1 为无效填充。"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # pin_memory + non_blocking：H2D 拷贝可与 CPU 端后续工作重叠
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """拼 prefill 输入：多条序列首尾相接成一维，用 cu_seqlens 标记边界。

        Q 和 K 的长度会不一样：Q 只含本轮要算的新 token，K 要覆盖整段上下文
        （含前缀命中和之前分块算过的部分），所以维护两套 cu_seqlens。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]    # 各序列 Q 在拼接结果中的起止前缀和
        cu_seqlens_k = [0]    # 各序列 K 的有效长度前缀和
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []     # 每个新 token 的 K/V 该写进池里哪个槽位
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end    # 从头算起：新 Q 要能看到前面所有上下文
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))    # 绝对位置，RoPE 要用
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 把 [start, end) 这段逻辑位置翻译成池里的绝对槽号，逐块处理
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size    # 首块可能从中间接着写
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size    # 末块只写到 end
                slot_mapping.extend(range(slot_start, slot_end))
        # K 比 Q 长说明有序列带着历史 KV，Attention 得靠块表去池里读
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 这些元信息通过全局 Context 传给各层，避免逐层透传参数
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """拼 decode 输入：每条序列只送 1 个 token，batch 维就是序列数。"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)     # 上轮采样出的那个 token
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))        # 有效上下文长度，防止读到末块空位
            # 新 token 一定落在末块里，偏移就是末块已用长度减一
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """走 eager 还是 CUDA Graph。

        prefill 每轮形状都不同，无法复用图；batch 超过捕获过的最大档（512）也
        只能回退 eager。其余 decode 情形挑一档够大的图，填好输入后重放。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]    # 选最小的够用档
            graph_vars = self.graph_vars
            # 图捕获时绑定的是这些固定缓冲区的地址，所以只能原地写入，不能换张量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)    # 填充位设 -1，Triton 写 KV 时会跳过
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()      # 填充位长度设 0，Attention 读不到脏数据
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # 图里只有模型主干，logits 和采样仍在图外按真实 bs 计算
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """一轮完整执行：准备输入 → 前向 → 采样。非 rank0 不产出 token。"""
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # TP 下 logits 被 gather 到 rank0，只有它能采样；其余 rank 返回 None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """给若干档 batch size 各捕获一张 decode 图，运行时按需挑一档重放。

        图只覆盖 self.model(...) 这段主干，compute_logits 和采样在图外。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 所有档共用这一组固定缓冲区；重放前往里面写数据，重放后从 outputs 取结果
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))    # 小 batch 密、大 batch 疏
        self.graphs = {}
        self.graph_pool = None

        # 从最大档开始捕获：显存池先按最大需求建好，后面的小图直接复用，不再新申请
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
