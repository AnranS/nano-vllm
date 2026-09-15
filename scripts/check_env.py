"""Check the CUDA development environment and optionally run a local Qwen3 model."""

import argparse
import atexit
from importlib.metadata import version
from pathlib import Path
import sys


def check_kernels():
    """校验两个外部算子和项目自带的 Triton kernel 算得对不对。

    torch / flash_attn 在函数内部导入是有意为之：main() 会先打印 Python 和各包
    版本，导入失败时这些诊断信息已经输出，便于定位是哪一层装错了；
    `--help` 也不会因为缺 CUDA 而起不来。
    """
    import torch
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from nanovllm.layers.attention import store_kvcache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and GPU access.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch CUDA: {torch.version.cuda}")

    # 1) prefill 路径：拿 PyTorch 自带的 SDPA 当参考答案，比对 FlashAttention
    torch.manual_seed(0)
    q, k, v = [torch.randn(4, 2, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
    lengths = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(q, k, v, lengths, lengths, 4, 4, causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    # 2) 写入路径：本项目的 Triton kernel 把 K/V 按槽位写进池，读回来应逐元素相等
    k_cache = torch.zeros(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.arange(4, device="cuda", dtype=torch.int32)
    store_kvcache(k, v, k_cache, v_cache, slots)
    torch.testing.assert_close(k_cache[0, :4], k)
    torch.testing.assert_close(v_cache[0, :4], v)
    # 3) decode 路径：只送最后一个 Q，走分页 kvcache 内核，结果应与 prefill 的末行一致
    decoded = flash_attn_with_kvcache(
        q[-1:].unsqueeze(0), k_cache, v_cache,
        cache_seqlens=torch.tensor([4], device="cuda", dtype=torch.int32),
        block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
        causal=True,
    )
    torch.testing.assert_close(decoded[0, 0], actual[-1], atol=5e-3, rtol=5e-3)
    torch.cuda.synchronize()
    print("PASS: FlashAttention prefill/decode and Triton KV cache kernels")


def check_model(path, cuda_graph):
    """跑一次最小规模的真实推理。同样延迟导入，理由见 check_kernels。"""
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply briefly: what is 1 + 1?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # 刻意压小：512 上下文、2 条序列、40% 显存，验证机上也能跑，不跟别的进程抢卡
    llm = LLM(
        str(path), enforce_eager=not cuda_graph, tensor_parallel_size=1,
        max_model_len=512, max_num_batched_tokens=512, max_num_seqs=2,
        gpu_memory_utilization=0.4,
    )
    try:
        output = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=32), use_tqdm=False)[0]
        if not output["token_ids"] or not output["text"].strip():
            raise RuntimeError("The model did not generate text.")
        print(f"PASS: model inference ({'CUDA Graph' if cuda_graph else 'eager'})")
        print(f"Output: {output['text']}")
    finally:
        # LLMEngine 在构造时注册了 atexit 清理；这里主动收尾，注销掉避免重复执行
        atexit.unregister(llm.exit)
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local Qwen3 model directory (optional)")
    parser.add_argument("--cuda-graph", action="store_true", help="Also exercise CUDA Graph during model inference")
    args = parser.parse_args()
    if args.cuda_graph and args.model is None:
        parser.error("--cuda-graph requires --model")
    if args.model is not None and not args.model.expanduser().is_dir():
        parser.error("--model must point to an existing local model directory")

    # 先打版本再碰 GPU：装错版本时这几行就是最有用的线索
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for package in ("nano-vllm", "torch", "triton", "transformers", "flash-attn", "xxhash"):
        print(f"{package}: {version(package)}")
    check_kernels()
    if args.model is not None:
        check_model(args.model.expanduser().resolve(), args.cuda_graph)


if __name__ == "__main__":
    main()
