from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """采样配置。这里只做最小集合，没有 top-k / top-p / 惩罚项。"""
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False    # True 则一直生成到 max_tokens，压测时用来固定输出长度

    def __post_init__(self):
        # Sampler 要拿温度做除法，取 0 会溢出；贪心解码需要单独实现，这里直接禁掉
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
