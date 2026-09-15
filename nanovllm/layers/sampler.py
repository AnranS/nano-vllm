import torch
from torch import nn


class Sampler(nn.Module):
    """按温度做多项式采样。"""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 升到 fp32 再除温度：低精度下 softmax 容易丢掉长尾概率
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 指数竞赛技巧：argmax(p_i / E_i)，E_i ~ Exp(1)，结果等价于按 p 采样。
        # 比 torch.multinomial 更适合放进 CUDA Graph / torch.compile——纯逐元素
        # 运算加一次 argmax，没有依赖动态形状的内核。clamp 防除零。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
