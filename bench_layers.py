"""对比 nn.Module 版与无模块版的 layers，量出 nn.Module 这层包装的成本。

两套实现共享同一批权重张量（layers_without_torch.adopt 直接接管，不复制），
所以 GPU 上执行的 kernel 完全相同，差异只在 Python 调用路径：
  - nn.Module.__call__ → _call_impl 的 hook 检查
  - self.weight 经 nn.Module.__getattr__ 去 _parameters 字典里找
  - nn.ModuleList 迭代走 _modules 字典

用法：
    python bench_layers.py                        # 默认开 torch.compile，与线上一致
    TORCHDYNAMO_DISABLE=1 python bench_layers.py  # 关掉编译，隔离出纯 Python 开销
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model

from layers_without_torch import adopt, bind_kv_cache, qwen3_compute_logits, qwen3_forward
from layers_without_torch.qwen3 import qwen3_decoder_layer as f_decoder_layer
from layers_without_torch.embed_head import embedding as f_embedding
from layers_without_torch.layernorm import rms_norm as f_rms_norm
from layers_without_torch.linear import linear as f_linear, row_linear as f_row_linear
from layers_without_torch.rotary_embedding import apply_rope as f_apply_rope
from layers_without_torch.sampler import sample as f_sample
from layers_without_torch.activation import silu_and_mul as f_silu_and_mul

PATH = "/home/codex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
BLOCK_SIZE = 256
NUM_BLOCKS = 72


def _loop(fn, iters):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def compare(fa, fb, iters=200, warmup=50, repeats=5):
    """A/B 对测，各返回单次调用耗时（秒）。

    三个刻意的选择：
    1. 循环内不同步。eager decode 下 CPU 是瓶颈、GPU 队列始终非空，测到的就是
       CPU 侧发起工作的耗时——正是两套实现唯一有差别的地方。
    2. 交替测。先把 A 测完再测 B 的话，GPU 频率、显存碎片、缓存状态都漂了；
       实测同一组配置两次跑能差出一倍的结论。交替之后两边处境相同。
    3. 各取多轮最小值。延迟分布是单边的——只会被其他进程和降频拖慢，不会莫名
       变快，所以最小值比平均值稳得多。
    """
    for _ in range(warmup):
        fa()
        fb()
    ba = bb = float("inf")
    for _ in range(repeats):
        ba = min(ba, _loop(fa, iters))
        bb = min(bb, _loop(fb, iters))
    return ba, bb


def setup():
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    hf_config = AutoConfig.from_pretrained(PATH)
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(hf_config)
    load_model(model, PATH)
    sampler = Sampler()

    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    # 必须清零：线上的 KV 池只会读到写过的位置，这里的假上下文却会读到整块。
    # 用 torch.empty 的话 attention 会算到未初始化内存里的 NaN，正确性对比失去意义。
    kv_cache = torch.zeros(2, hf_config.num_hidden_layers, NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_dim)
    layer_id = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]
            layer_id += 1

    free = adopt(model)
    bind_kv_cache(free, kv_cache)
    return hf_config, model, free, sampler


def decode_context(bs, cached_len):
    """bs 条序列各占一个块，各自已缓存 cached_len 个 token，本轮生成第 cached_len+1 个。"""
    slot_mapping = torch.tensor([i * BLOCK_SIZE + cached_len for i in range(bs)], dtype=torch.int32)
    context_lens = torch.full((bs,), cached_len + 1, dtype=torch.int32)
    block_tables = torch.tensor([[i] for i in range(bs)], dtype=torch.int32)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)


def prefill_context(seq_len):
    """单条序列的整段 prefill，无前缀命中。"""
    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    slot_mapping = torch.arange(seq_len, dtype=torch.int32)
    set_context(True, cu, cu, seq_len, seq_len, slot_mapping, None, None)


def row(name, module_s, free_s, unit=1e6, prec=1):
    gain = (module_s - free_s) / module_s * 100
    print(f"  {name:26} {module_s * unit:9.{prec}f}  {free_s * unit:9.{prec}f}  "
          f"{(module_s - free_s) * unit:+9.{prec}f}  {gain:+6.1f}%")


def bench_ops(hf_config, model, free, sampler, n):
    """逐算子对比。n 是本轮 token 数：decode 时等于 batch size。"""
    layer = model.model.layers[0]
    fl = free.model.layers[0]
    h = hf_config.hidden_size
    dtype = hf_config.dtype
    x = torch.randn(n, h, dtype=dtype)
    resid = torch.randn(n, h, dtype=dtype)

    print(f"\n[逐算子 · {n} token]              nn.Module     无模块        节省      比例")
    print(f"  {'':28} {'µs/call':>9}  {'µs/call':>9}  {'µs':>9}")

    row("RMSNorm", *compare(lambda: layer.input_layernorm(x),
                            lambda: f_rms_norm(fl.input_layernorm, x)))
    row("RMSNorm + residual", *compare(lambda: layer.input_layernorm(x, resid),
                                       lambda: f_rms_norm(fl.input_layernorm, x, resid)))

    gate_up = torch.randn(n, hf_config.intermediate_size * 2, dtype=dtype)
    row("SiluAndMul", *compare(lambda: layer.mlp.act_fn(gate_up),
                               lambda: f_silu_and_mul(gate_up)))

    attn = layer.self_attn
    q = torch.randn(n, attn.num_heads, attn.head_dim, dtype=dtype)
    k = torch.randn(n, attn.num_kv_heads, attn.head_dim, dtype=dtype)
    pos = torch.arange(n, dtype=torch.int64)
    row("RoPE", *compare(lambda: attn.rotary_emb(pos, q, k),
                         lambda: f_apply_rope(fl.self_attn.rope, pos, q, k)))

    row("Linear (qkv_proj)", *compare(lambda: attn.qkv_proj(x),
                                      lambda: f_linear(fl.self_attn.qkv_proj, x)))
    o = torch.randn(n, attn.num_heads * attn.head_dim, dtype=dtype)
    row("RowLinear (o_proj)", *compare(lambda: attn.o_proj(o),
                                       lambda: f_row_linear(fl.self_attn.o_proj, o)))

    ids = torch.randint(0, hf_config.vocab_size, (n,), dtype=torch.int64)
    row("Embedding", *compare(lambda: model.model.embed_tokens(ids),
                              lambda: f_embedding(free.model.embed_tokens, ids)))

    logits = torch.randn(n, hf_config.vocab_size, dtype=torch.float32)
    temps = torch.full((n,), 0.6, dtype=torch.float32)
    row("Sampler", *compare(lambda: sampler(logits, temps),
                            lambda: f_sample(logits, temps)))


def bench_model(hf_config, model, free, n, is_prefill):
    ids = torch.randint(0, hf_config.vocab_size, (n,), dtype=torch.int64)
    if is_prefill:
        prefill_context(n)
        pos = torch.arange(n, dtype=torch.int64)
    else:
        decode_context(n, 128)
        pos = torch.full((n,), 128, dtype=torch.int64)

    m, f = compare(lambda: model(ids, pos), lambda: qwen3_forward(free, ids, pos),
                   iters=30, warmup=10)
    reset_context()
    row(f"prefill {n} token" if is_prefill else f"decode bs={n}", m, f, unit=1e3, prec=2)


def capture(fn, ids, pos):
    """按 ModelRunner.capture_cudagraph 的做法捕获一张图：先预热，再录制。"""
    fn(ids, pos)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(ids, pos)
    torch.cuda.synchronize()
    return graph


def bench_graph(hf_config, model, free, n):
    ids = torch.randint(0, hf_config.vocab_size, (n,), dtype=torch.int64)
    pos = torch.full((n,), 128, dtype=torch.int64)
    decode_context(n, 128)

    gm = capture(lambda i, p: model(i, p), ids, pos)
    gf = capture(lambda i, p: qwen3_forward(free, i, p), ids, pos)
    m, f = compare(gm.replay, gf.replay, iters=100, warmup=20)
    reset_context()
    row(f"decode bs={n}（图重放）", m, f, unit=1e3, prec=3)
    del gm, gf


def check(hf_config, model, free, compiled):
    """两套实现必须等价，否则后面的速度对比没有意义。

    关掉 torch.compile 时要求逐位相同——这才是「逻辑是否一致」的判据。
    开着编译时允许极小漂移：两套实现是两个不同的 code object，各自独立编译，
    融合和归约顺序可能不同，1 ULP 的差在 28 层 bf16 里会累积成肉眼可见的数。
    所以此时只校验相对量级和 argmax。
    """
    n = 64
    ids = torch.randint(0, hf_config.vocab_size, (n,), dtype=torch.int64)
    pos = torch.full((n,), 128, dtype=torch.int64)

    decode_context(n, 128)
    h = model.model.embed_tokens(ids)
    worst = (h.float() - f_embedding(free.model.embed_tokens, ids).float()).abs().max().item()
    h_m, r_m = h.clone(), None
    h_f, r_f = h.clone(), None
    for src, dst in zip(model.model.layers, free.model.layers):
        h_m, r_m = src(pos, h_m, r_m)
        h_f, r_f = f_decoder_layer(dst, pos, h_f, r_f)
        worst = max(worst, (h_m.float() - h_f.float()).abs().max().item())
        # 下一层喂同一个输入，避免误差累积掩盖掉单层的真实差异
        h_f, r_f = h_m.clone(), r_m.clone()

    a = model(ids, pos)
    b = qwen3_forward(free, ids, pos)
    la = model.compute_logits(a)
    lb = qwen3_compute_logits(free, b)
    reset_context()

    e2e = (la.float() - lb.float()).abs().max().item()
    scale = la.float().abs().max().item()
    agree = (la.argmax(-1) == lb.argmax(-1)).all().item()
    print(f"正确性：逐层最大差 {worst:.3e}；端到端 logits 最大差 {e2e:.3e}"
          f"（相对量级 {e2e / scale:.2e}），argmax 全一致 {bool(agree)}")
    if compiled:
        # argmax 一致才是真判据；这条只是拦住量级明显跑飞的情况。
        assert e2e / scale < 0.15, "端到端漂移超出编译噪声的合理范围"
    else:
        assert worst == 0.0 and e2e == 0.0, "未开编译时结果就不一致，说明实现有逻辑差异"
    assert agree, "两套实现预测出的 token 不一致"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    compiled = os.environ.get("TORCHDYNAMO_DISABLE", "0") != "1"
    print(f"=== nn.Module vs 无模块 === torch.compile: {'on' if compiled else 'off'}")
    hf_config, model, free, sampler = setup()

    # 全程 inference_mode：线上就是这么跑的，而且 nn.Parameter 默认 requires_grad，
    # 不关掉的话 nn.Module 侧会白白背上 autograd 记账，对比就不公平了。
    with torch.inference_mode():
        check(hf_config, model, free, compiled)

        for n in (1, 64):
            bench_ops(hf_config, model, free, sampler, n)

        print(f"\n[整模型前向]                     nn.Module     无模块        节省      比例")
        print(f"  {'':28} {'ms/call':>9}  {'ms/call':>9}  {'ms':>9}")
        bench_model(hf_config, model, free, 1, False)
        bench_model(hf_config, model, free, 64, False)
        bench_model(hf_config, model, free, 512, True)

        if not args.skip_graph:
            print(f"\n[CUDA Graph]                     nn.Module     无模块        节省      比例")
            print(f"  {'':28} {'ms/call':>9}  {'ms/call':>9}  {'ms':>9}")
            bench_graph(hf_config, model, free, 1)
            bench_graph(hf_config, model, free, 64)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
