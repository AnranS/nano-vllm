import time
from random import randint, seed
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    """吞吐压测：用随机 token 造一批长短不一的请求，量端到端生成速度。

    输入是随机 token id 而非真实文本——这里只关心调度和执行的吞吐，
    生成内容是否通顺无关紧要。改用上面注释掉的 import 即可对比 vLLM。
    """
    seed(0)    # 固定随机数，保证每次跑的负载完全一致
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = snapshot_download("Qwen/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)    # 开 CUDA Graph 才是正常推理姿势

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # ignore_eos=True：不让 EOS 提前截断，输出长度完全由 max_tokens 决定，
    # 这样总 token 数是已知常量，吞吐才可比
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())    # 预热一次，把编译和图捕获排除在计时之外
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)    # 只统计输出 token，不含 prompt
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
