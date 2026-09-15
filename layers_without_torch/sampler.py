"""采样，去掉 nn.Module 包装。原版 Sampler 也是个无参数模块。"""
import torch


@torch.compile
def sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
