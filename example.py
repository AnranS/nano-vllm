from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """最小可跑示例：两条请求走一遍完整的 prefill + decode。

    第一次调试建议保持 enforce_eager=True 和 tensor_parallel_size=1，
    出问题时调用链最短；确认能生成之后再逐项打开优化。
    """
    path = "/home/codex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 引擎只收 token，不懂对话格式；chat template 要在这一层套好
    templated_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(templated_prompts, sampling_params)
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        text = tokenizer.decode(output["token_ids"], skip_special_tokens=True)
        # Qwen3 是混合推理模型，思考过程包在 <think></think> 里，这里拆开分别显示
        if text.startswith("<think>"):
            thinking, _, answer = text.removeprefix("<think>").partition("</think>")
        else:
            thinking, answer = "", text
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(prompts)}] {prompt}")
        print("-" * 60)
        if thinking:
            print(f"<thinking>\n{thinking.strip()}\n</thinking>\n")
        if answer.strip():
            print(answer.strip())
        print(f"\n({len(output['token_ids'])} tokens)")


if __name__ == "__main__":
    main()
