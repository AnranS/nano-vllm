from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """对外的名字，接口与 vLLM 的 LLM 对齐；实现全在 LLMEngine 里。"""
    pass
