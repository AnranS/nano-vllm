# nano-vLLM 学习文档

**[在线阅读](https://anrans.github.io/nano-vllm/)** · 部署在本 fork 的 GitHub Pages。

双击 `index.html`，即可在浏览器离线阅读。无需启动服务器、安装前端依赖或连接网络。

内容依据本地 nano-vLLM 提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 与生成时工作区源码编写；精确文件哈希在 `source/manifest.json` 中。

## 阅读顺序

1. 阅读路线：先明确能解释什么。
2. 一轮请求：`generate → schedule → run → postprocess`。
3. 调度：连续批处理、分块 Prefill、抢占。
4. KV 分页：显存池、块表、可交互地址映射。
5. 前缀复用：链式哈希、引用计数、空闲块仍可命中。
6. Attention：`slot_mapping` 写入与 `block_table` 读取。
7. CUDA Graph 与编译：减少重复发起计算的开销。
8. 多 GPU：张量分片、CPU 共享内存与 NCCL。
9. 动手实验：CPU 管理器实验、GPU 阅读观察点、自检。
10. 术语与功能地图。

每章带本地源码快照链接和真实行号。页面中的源码不是实时文件视图，代码变更后需要重新生成。

`nanovllm/` 下的源码带逐段中文注释，快照页面会一并显示。推荐读法是每章先读正文建立框架，再点进快照顺着注释走一遍代码。第 09 章有「源码文件 → 该读哪一章」的反向索引。

## CPU 教学实验

在项目根目录运行：

```sh
.venv/bin/python docs/web/cpu_lab.py
```

需要 `numpy` 和 `xxhash`。实验直接运行本地 Sequence、BlockManager 和 Scheduler；仅使用简单配置和人工 token 结果驱动状态，绕过包入口的 GPU 模块导入。它不会计算 K/V 向量，不代表完整模型推理，也不测性能。

本地 macOS 可做源码阅读与这些 CPU 实验。完整推理路径仍要求 CUDA / NCCL / Triton / FlashAttention，平台依赖条件不会自动提供 Mac GPU 后端。

## 维护文档

在项目根目录运行：

```sh
python3 docs/web/build.py
python3 docs/web/validate.py
```

生成器只使用 Python 标准库，要求 Python 3.10+。正文在 `content.py`，样式在 `style.css`，页内地址演示在 `app.js`。生成器更新 HTML、源码快照与引用行号；不会自动重写正文解释。算法变化后需人工复核相关章节。

`validate.py` 检查离线链接、锚点、源码哈希和 AST 符号行号，不执行浏览器，也不执行 GPU 推理。

## 启动本地服务

运行 `start.sh`，即可启动服务并自动打开浏览器。脚本优先使用项目的 `.venv/bin/python`，否则使用系统 `python3`；无须安装额外依赖。

在项目根目录运行：

```sh
./docs/web/start.sh
```

默认地址为 `http://127.0.0.1:8769/index.html`。保持启动脚本的终端运行，按 **Ctrl+C** 停止服务。重复启动会复用由该脚本启动的同一份文档服务；如果端口被其他服务占用，会提示换端口，不会终止其他进程。

可选参数：

```sh
./docs/web/start.sh --port 8770
./docs/web/start.sh --no-browser
```

Windows / Linux 或其他终端也可直接使用标准库入口：

```sh
python3 docs/web/serve.py
```

仍可双击 `index.html` 离线阅读，无须启动服务。

源码快照遵循原项目 MIT 许可，见 `source/LICENSE.txt`。文档未修改推理代码、项目依赖或现有锁文件。

## GitHub Pages 自动发布

工作流位于 `.github/workflows/deploy-docs.yml`。向 `main` 推送文档、模型源码或工作流的改动后，GitHub Actions 会重新生成页面与源码快照、检查离线链接和引用行号，再发布到 GitHub Pages。也可以在 Actions 页面手动运行此工作流。

发布内容为 HTML、CSS、JavaScript 和源码快照；网页运行不需要 Python、模型权重或 GPU。本地离线阅读与 `start.sh` 启动方式继续可用。
