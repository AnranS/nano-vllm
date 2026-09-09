# 开发环境

在项目根目录操作。使用 Python 3.12 和独立的 `.venv`，`uv sync` 会以可编辑模式安装本项目，修改 `nanovllm/` 后立即生效。

本机已验证（2026-09-09）：WSL2、RTX 4080 SUPER 16 GB、Python 3.12.3、PyTorch 2.12.1 / CUDA 13.0、Triton 3.7.1、Transformers 5.15.1。依赖一致性检查、CPU 实验、GPU 算子数值检查，以及 Qwen3-0.6B 的 eager / CUDA Graph 推理均通过。

## 安装

本配置用于 Linux / WSL2 上的 NVIDIA GPU，使用 CUDA 13.0 预编译依赖，需要支持 CUDA 13 的驱动。先确认 `nvidia-smi` 正常。若要开发、编译 CUDA 扩展，还需要 CUDA Toolkit；`nvcc --version` 应正常，`CUDA_HOME` 应指向包含 `bin/nvcc` 的 Toolkit 目录。

```bash
uv sync --locked --cache-dir .cache/uv
source .venv/bin/activate
```

`.python-version` 选择 Python 3.12，`uv.lock` 固定依赖版本。下载的包存放在已被 Git 忽略的 `.cache/uv/` 中，虚拟环境 `.venv/` 也不会提交。

`pyproject.toml` 将开发环境固定为 PyTorch 2.12.1，并从 Astral CUDA 13 索引安装 `flash-attn==2.8.3.post1+cu.13.0.torch.2.12`。该 wheel 明确要求 `torch==2.12.*`；索引设为 `explicit`，只用于 FlashAttention，其余包使用 PyPI。首次安装需要下载较大的 PyTorch / CUDA 运行库，后续同步会复用缓存。

原锁文件的 PyTorch 2.13.0 与 FlashAttention 2.8.3.post1 组合在本机源码编译失败，因此开发环境改用上述配套预编译版本。升级时需要同时核对 PyTorch、CUDA 和 FlashAttention wheel 的版本。

依据：[uv 官方预编译扩展配置](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-gpu-enabled-pytorch-extensions)、[Astral CUDA 13 FlashAttention 索引](https://wheels.astral.sh/simple/cu130/flash-attn/)。

## 验证

```bash
# 检查已安装包之间的依赖是否一致
uv pip check --python .venv/bin/python --cache-dir .cache/uv

# 执行 FlashAttention prefill/decode 和项目自身的 Triton KV Cache kernel
python scripts/check_env.py

# 运行已有的 CPU 调度、前缀缓存实验
python docs/web/cpu_lab.py
```

`check_env.py` 会将 GPU 运算结果与参考结果进行比较；需要能访问 GPU 的终端。若在受限沙箱中出现 GPU 不可访问，请先在普通 WSL / Linux 终端检查 `nvidia-smi`。

已有本地 Qwen3 模型时，可进一步验证完整推理：

```bash
python scripts/check_env.py --model /path/to/Qwen3-0.6B
python scripts/check_env.py --model /path/to/Qwen3-0.6B --cuda-graph
```

验证脚本限制上下文为 512 token、最多 2 个序列、显存预算为总显存的 40%，生成最多 32 token。默认使用 eager 执行；`--cuda-graph` 会额外验证 CUDA Graph 路径。它使用本地模型目录，不下载权重。

本机已有 Qwen3-0.6B 缓存，可以直接运行：

```bash
NANOVLLM_MODEL=/home/codex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
python scripts/check_env.py --model "$NANOVLLM_MODEL"
```

`example.py` 已直接使用上面的本机模型缓存目录，激活环境后运行 `python example.py` 即可，无需创建软链接。若更换模型目录，修改 `example.py` 中的 `path`。

## 日常使用

每次打开新终端：

```bash
cd /home/codex/nano-vllm
source .venv/bin/activate
```

也可以直接使用 `.venv/bin/python`。学习文档服务可通过 `./docs/web/start.sh` 启动，文档维护方式见 [学习文档 README](web/README.md)。

更新依赖后重新运行 `uv sync` 和环境验证。不要在这个项目中使用其他项目虚拟环境里的 `pip`，以免把包装到错误位置。
