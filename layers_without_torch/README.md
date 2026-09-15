# layers_without_torch —— 拆掉 nn.Module 的一份 layers

## 这是什么

`nanovllm/layers/` 和 `nanovllm/models/qwen3.py` 的等价实现，区别只有一处：**不用 `nn.Module`**。

张量还是 `torch.Tensor`，矩阵乘还是 cuBLAS，attention 还是 flash-attn，Triton 的 KV 写入 kernel 直接复用原版。
换掉的只是 Python 这一侧的组织方式：

| | `nanovllm/layers/` | 这里 |
|---|---|---|
| 权重 | `nn.Parameter`，存在 `self._parameters` | 裸 `Tensor`，存在 `__slots__` 里 |
| 前向 | `Module.__call__` → `_call_impl` → `forward` | 直接调模块级函数 |
| 子层容器 | `nn.ModuleList` | `tuple` |

> 文件夹名沿用了最初的叫法。更准确的描述是 "layers without nn.Module"——torch 并没有被拿掉。

## 为什么值得单独测

`nn.Module` 在推理时做了三件对结果毫无贡献的事，每次调用都要重做一遍：

1. **`_call_impl` 的 hook 检查**。每次调用都要查 8 个 hook 字典是不是空的，才走到 `forward`。
2. **参数访问走 `__getattr__`**。`nn.Parameter` 不在实例的 `__dict__` 里，所以 `self.weight` 的常规属性查找一定会落空，
   然后 fallback 到 `nn.Module.__getattr__`，依次翻 `_parameters` / `_buffers` / `_modules` 三个字典。
   一次 `forward` 里每访问一次权重就付一次。
3. **`nn.ModuleList` 迭代**走 `_modules` 字典而不是普通列表。

跑一次 Qwen3-0.6B 前向要触发约 **395 次模块调用**（28 层 × 13 + 首尾若干），
每次几微秒，累起来就是毫秒级——而这段时间 GPU 完全是空的。

## 怎么跑

在仓库根目录：

```bash
python bench_layers.py                        # 和线上一致：开 torch.compile
TORCHDYNAMO_DISABLE=1 python bench_layers.py  # 关掉编译，隔离出纯 Python 开销
python bench_layers.py --skip-graph           # 跳过 CUDA Graph 那一组
```

`bench_layers.py` 会先做正确性校验再计时。两套实现共享同一批权重张量
（`adopt()` 直接接管，不复制），所以 GPU 上跑的 kernel 完全相同，测出来的差异只可能来自 Python 调用路径。

## 结果

RTX 4080 SUPER / Qwen3-0.6B / CUDA 13.0 / PyTorch 2.12.1。下面是 `TORCHDYNAMO_DISABLE=1` 那一组，
因为不开编译时 dynamo 的 guard 开销不会盖住要测的东西，读数最干净。

**正确性**：逐层最大差 `0.000e+00`，端到端 logits 最大差 `0.000e+00`。完全逐位相同。

**单算子**（µs/call，节省量基本与张量大小无关）

| 算子 | nn.Module | 无模块 | 节省 |
|---|---|---|---|
| Linear (qkv_proj) | 13.9 | 11.8 | +2.0 µs (+14.7%) |
| RowLinear (o_proj) | 15.1 | 12.9 | +2.2 µs (+14.4%) |
| Embedding | 16.8 | 14.5 | +2.3 µs (+13.4%) |
| SiluAndMul | 28.2 | 25.3 | +2.9 µs (+10.1%) |
| RMSNorm + residual | 115.8 | 114.7 | +1.1 µs (+1.0%) |
| RoPE | 213.6 | 207.4 | +6.2 µs (+2.9%) |

省下的是**每次调用固定约 2 µs**，和张量多大无关——这正是「纯 CPU 侧开销」的特征。
所以占比取决于分母：单 kernel 的 Linear 上是 14%，而 RMSNorm、RoPE 这种要发好几个 kernel 的算子，
2 µs 摊到上百微秒里就看不见了。

**整模型前向**（ms/call）

| 场景 | nn.Module | 无模块 | 节省 |
|---|---|---|---|
| eager · decode bs=1 | 27.47 | 25.87 | +1.60 ms (+5.8%) |
| eager · decode bs=64 | 27.76 | 25.81 | +1.95 ms (+7.0%) |
| eager · prefill 512 token | 30.62 | 28.08 | +2.54 ms (+8.3%) |
| **CUDA Graph · decode bs=1** | **3.751** | **3.752** | **-0.001 ms (-0.0%)** |
| **CUDA Graph · decode bs=64** | **6.055** | **6.057** | **-0.002 ms (-0.0%)** |

开着 `torch.compile` 跑结论一样：eager 省 6~8%，图重放 0%。

## 就省这么点？——把 26 ms 拆开看

这是看到 6% 时必然会问的问题。答案是：**拆掉 `nn.Module` 的收益结构上就封顶了**，因为它不改变算子派发的次数。

数一下一次 decode 前向走了多少次 torch 的 Python 层调用（用 `TorchFunctionMode` 统计，关编译）：

```
nn.Module 版   3,090 次
无模块版       3,090 次   ← 一次都没少
```

再测单次派发的地板——最廉价的 torch 算子，1 个元素的原地加法：**9.00 µs/call**。

乘回去：`3,090 次 × 9.00 µs = 27.8 ms`，而实测整个 eager 前向是 `26.19 ms`。
**这 26 ms 几乎全部就是派发成本本身**，GPU 算了多少完全不影响总时间。
（在模型里平摊到每次调用是 8.48 µs，比孤立测的 9.00 µs 略低，因为连续调用能流水起来。）

于是优化的阶梯长这样（decode bs=64）：

| 配置 | ms | 相对基线 |
|---|---|---|
| eager，无编译 | 27.76 | 1.0× |
| 拆掉 `nn.Module` | 25.81 | 1.08× |
| 再加 `torch.compile` | 18.96 | 1.46× |
| 再加 CUDA Graph | 3.92 | **7.1×** |

`nn.Module` 是最矮的一级。要真的快只有两条路：**减少 Python→C++ 的穿越次数**
（`torch.compile` 把多个算子融成一个 kernel），或者**彻底不走 Python**
（CUDA Graph 在捕获时录下 kernel launch 序列，replay 时一次性提交，派发次数归零）。

## 结论

**eager 模式能省 6~8%，CUDA Graph 下省 0%。**

后半句才是重点。图重放时这些 Python 代码根本不执行——捕获阶段已经把 kernel launch 序列录进了图里，
replay 是驱动层一次性提交整串 GPU 操作。`nn.Module` 的开销、`torch.compile` 的 guard 开销、
甚至整个 Python 解释器，在 replay 期间都不在路径上。这也是为什么图重放能从 27 ms 掉到 3.8 ms：
它消掉的是**全部** CPU 侧开销，其中 `nn.Module` 只占 1.6 ms 那一小块。

对应到 nano-vLLM 实际怎么跑：

- **decode**（默认走 CUDA Graph）：**0 收益**。
- **prefill**（形状每轮都变，`ModelRunner.run_model` 只在 decode 且 batch ≤ 512 时才用图，prefill 恒走 eager）：**约 8%**。
- **`enforce_eager=True` 时**（`example.py` 的默认配置，方便调试）：全程约 6~8%。

端到端到底提多少？跑一遍 `bench.py` 的负载（256 条请求、输入输出各 100~1024 token），
在 `ModelRunner.run` 上按分支计时：

| 路径 | 轮数 | 耗时 | 占模型执行 |
|---|---|---|---|
| prefill（eager） | 136 | 4.80 s | **14.2%** |
| decode（CUDA Graph） | 1938 | 29.04 s | **85.8%** |

**CUDA Graph 覆盖了 85.8% 的执行时间，而那部分收益恰好是 0。**
所以端到端收益上限 = 8% × 14.2% ≈ **1.1%**。
**结论是：`nn.Module` 的开销真实存在且可测，但 nano-vLLM 已经用 CUDA Graph 把它连同其它 CPU 开销一起绕过去了。**
真正值得记住的不是「拆掉 nn.Module 能提速」，而是「CUDA Graph 到底替你省掉了什么」。

## 一些测量上的坑

写这个 bench 时踩到的，都反映在代码里：

- **KV 池必须清零**。最初用 `torch.empty` 分配，假上下文让 attention 读到了整块未初始化内存，
  里面残留着上一次分配的大数值，喂进 softmax 后把 1 ULP 的差放大到肉眼可见，正确性对比直接失效。
- **A/B 要交替测**。先把 A 测完再测 B，GPU 频率和缓存状态已经漂了；同一组配置两次跑能得出
  +17.5% 和 +7.5% 两个结论。交替之后稳定在 6~8%。
- **取最小值而不是平均值**。延迟只会被其他进程和降频拖慢，不会莫名变快。
- **全程 `torch.inference_mode()`**。`nn.Parameter` 默认 `requires_grad=True`，不关掉的话 nn.Module 侧
  会白白背上 autograd 记账，对比就不公平了（线上 `ModelRunner` 本来也是全程 inference_mode）。
- **开着 `torch.compile` 时两边结果不逐位相同**。两套实现是两个不同的 code object，各自独立编译，
  融合和归约顺序可能不同，1 ULP 的差在 28 层 bf16 里会累积到 `1e-2` 相对量级（argmax 仍然一致）。
  所以判定「逻辑是否等价」必须关掉编译看——关掉后是严格的 0。
