import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """初始化引擎：配置、多卡 worker、rank0 runner、tokenizer、调度器。"""
        # 只保留 Config 认识的字段，丢掉无关 kwargs，避免构造报错
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 所有 Sequence 共用同一块大小（与 KV cache 分页一致）
        Sequence.block_size = config.kvcache_block_size
        # 张量并行时，rank1..N-1 的子进程与同步事件
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")  # CUDA 多进程必须用 spawn，不能用 fork
        # 启动其余 GPU 上的 ModelRunner worker（rank0 在主进程）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank0：加载模型、分配 KV cache；events 用来指挥其他 worker
        self.model_runner = ModelRunner(config, 0, self.events)
        # 分词器；并把 EOS 写回 config，供 scheduler 判断结束
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 进程退出时自动清理 worker，避免僵尸进程占 GPU
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 调度：决定本轮算哪些序列，以及是 prefill 还是 decode
        # seqs = 本轮入批的序列；is_prefill=True 表示还在吃 prompt，False 表示逐 token 生成
        seqs, is_prefill = self.scheduler.schedule()
        # 给 generate 用的吞吐计数：prefill 用正数（本轮处理的 token 总数），
        # decode 用负数（-序列数，每条本轮只生成 1 个 token）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 真正跑模型：前向 + 采样，返回本轮每条序列新生成的 1 个 token id
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 收尾：追加 token、更新 KV 块哈希；若碰到 eos / 达到 max_tokens 则标记 FINISHED 并释放块
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只把本轮刚结束的序列交给上层；未结束的留在 running，下轮继续
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],          # 一批请求：文本或已 tokenize 的 token id
        sampling_params: SamplingParams | list[SamplingParams],  # 一份参数套全部，或每条各一份
        use_tqdm: bool = True,                         # 是否显示进度条
    ) -> list[str]:
        # 创建进度条：总数=请求数；disable=not use_tqdm 时关闭
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 若只传了一个 SamplingParams，复制成与 prompts 等长的列表，方便后面 zip 配对
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 逐条入队：tokenize → Sequence → 丢进 scheduler.waiting
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 用 seq_id 收集已完成序列的生成结果（可能乱序完成，所以先用 dict）
        outputs = {}
        # 进度条上显示的吞吐；初始为 0，等有 step 结果后再更新
        prefill_throughput = decode_throughput = 0.
        # 主循环：waiting/running 都空了才结束
        while not self.is_finished():
            t = perf_counter()                         # 本轮 step 计时起点
            output, num_tokens = self.step()           # schedule → run → postprocess；返回本轮完成的序列
            # step 用正负号编码阶段：>0 是 prefill（值为处理的 token 数），<=0 是 decode（-序列数）
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            # 刷新进度条后缀里的 Prefill / Decode 吞吐
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            # 本轮刚结束的序列写入 outputs，并推进进度条
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 按 seq_id 排序，保证返回顺序与输入 prompts 一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码成文本，同时保留 token_ids，方便调用方自己再 decode
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
