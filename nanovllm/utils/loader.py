import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """按名字把 safetensors 里的权重逐个灌进模型。

    只做两件事：处理融合模块的改名（q/k/v → qkv_proj），以及把切分逻辑
    委托给参数自带的 weight_loader（见 LinearBase）。加载器本身不知道
    张量并行怎么切。safe_open 以 "cpu" 打开，逐个张量读取，不整文件载入内存。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 融合权重：改名到目标模块，并告诉 loader 这是第几段
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # for-else：没有任何映射命中，说明是普通权重，按原名加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
