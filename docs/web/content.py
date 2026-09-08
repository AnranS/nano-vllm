"""Chinese chapters. Source citations resolve against the local Python AST."""
from html import escape

E = 'nanovllm/engine/'
L = 'nanovllm/layers/'


def code(text, caption='教学示意'):
    return f'<figure class="code-box"><figcaption>{caption}</figcaption><pre><code>{escape(text.strip())}</code></pre></figure>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th scope="col">{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def checkpoint(question, answer):
    return f'<details class="checkpoint"><summary>自检 · {question}</summary><div><p>{answer}</p></div></details>'


def flow(nodes):
    return '<div class="flow" aria-label="执行流程">' + '<span class="arrow" aria-hidden="true">→</span>'.join(f'<div class="node {kind}"><strong>{name}</strong><small>{detail}</small></div>' for name, detail, kind in nodes) + '</div>'


def chapter(file, label, title, desc, body):
    return dict(file=file, label=label, title=title, desc=desc, body=body)


def chapters(c):
    pages = []
    pages.append(chapter('index.html', '阅读路线', '从一个请求读懂 nano-vLLM', '先串起单 GPU 的一次生成，再逐步补上缓存、调度与执行优化。每章只解决一个问题。', f'''
<div class="takeaway"><strong>第一阶段的目标</strong><p>能解释：两个长度不同的请求怎么一起生成；其中一个结束后，另一个为什么还能继续；它们的 KV 存在什么地方。</p></div>
<h2>先建立这一条主线</h2>
{flow([('LLMEngine','串起每一轮执行',''),('Scheduler','选请求、安排 token、分块',''),('ModelRunner','准备输入、调用 GPU','gpu'),('postprocess','更新状态，回收请求','')])}
<p>CPU 管理请求、块号与执行顺序；GPU 保存权重和 KV 向量，完成张量运算。<code>ModelRunner</code> 本身仍是 CPU 上执行的 Python 对象。</p>
<h2>推荐的阅读节奏</h2>
<ol class="path-list"><li><strong>第一遍 · 01—03</strong><small>请求主循环 → 调度器 → KV 分页。先能手算状态，不追 CUDA 内核。</small></li><li><strong>第二遍 · 04—05</strong><small>前缀命中 → K/V 写入 → Attention 读取。把 CPU 块表和 GPU 数据接起来。</small></li><li><strong>第三遍 · 06—07</strong><small>CUDA Graph / 编译 → 多 GPU 张量并行与通信。理解每项优化减少什么开销。</small></li><li><strong>读完就做 · 08</strong><small>运行 CPU 小实验，验证分配、复用、回收和分块调度。</small></li></ol>
<h2>只需要这些前置知识</h2>
<p>认识 Python 类、列表、字典和队列；知道 Transformer 从输入计算 Q/K/V，再预测下一个 token；理解 Prefill 处理输入、Decode 逐轮生成。遇到术语时查最后一章，不必先学完整套 CUDA。</p>
<h2>本书对应哪个版本</h2>
<p>本地提交 <code>{c.commit[:12]}</code>。所有“查看源码”链接都打开随文档生成的本地快照，带真实行号，离线也能对照。提交之后的本地文件修改，以快照清单中的哈希为准。</p>
<p>这份工作区的 <code>pyproject.toml</code> 为 Triton / FlashAttention 添加了 Linux 平台条件。它便于在 Mac 上安装部分依赖，但推理代码仍直接使用 CUDA、NCCL 和 Triton，<strong>不等于支持 macOS GPU 推理</strong>。阅读网页无需任何 Python 或 GPU；CPU 实验也不运行模型。</p>
<p class="small">基础入口：{c.ref(E+'llm_engine.py','LLMEngine.step')} · {c.ref('nanovllm/config.py','Config')}</p>
'''))
    pages.append(chapter('01-request.html', '一轮请求', '01 / 一个请求怎样生成文本', '阅读 llm_engine.py 和 sequence.py。先理解每轮状态，再进入模型内部。', f'''
<h2>入口只有三件事</h2>
<p><code>generate()</code> 先把这一批 Prompt 加入等待队列，随后反复调用 <code>step()</code>，最后把生成的 token 解码成文本。一次 <code>step()</code> 是一次调度迭代，不是一个完整请求。</p>
{c.snippet(E+'llm_engine.py','LLMEngine.step')}
<p><code>schedule()</code> 决定本轮算谁、算多少 token；<code>run()</code> 得到预测 token；<code>postprocess()</code> 把结果写回请求状态。</p>
<h2>Sequence 是请求的账本</h2>
{table(['字段','含义'], [('token_ids / num_tokens','输入 token 加上已经采样出来的输出 token。'),('num_cached_tokens','已存在可用 KV 的 token 数，不包含刚采样出来但尚未送入模型的 token。'),('num_scheduled_tokens','本轮计划送入模型的 token 数。'),('block_table','请求的逻辑块 → GPU 缓存池块号。'),('status','WAITING → RUNNING → FINISHED；抢占时可退回 WAITING。')])}
<p>{c.ref(E+'sequence.py','Sequence')} · {c.ref(E+'scheduler.py','Scheduler.postprocess')}</p>
<h2>手推一遍：4 个输入，生成 3 个 token</h2>
<p>下面假设不中途遇到 EOS，<code>max_tokens=3</code>，不命中前缀，也不发生抢占。默认块大小仍是 256，4 个输入只是为了看清时间顺序。</p>
{table(['时刻','送入模型','模型计算后的有效 KV','采样后的总 token 数'], [('开始','—','0','4'),('第 1 轮 · Prefill','4 个输入 token','4','5'),('第 2 轮 · Decode','第 1 个输出 token','5','6'),('第 3 轮 · Decode','第 2 个输出 token','6；随后释放引用','7；请求结束')])}
<div class="takeaway"><strong>刚预测出的 token，还没有它自己的 KV。</strong><p>它要在下一轮作为模型输入，才产生 K/V。最后一个输出 token 如果已经触发结束，就不必再为它计算 KV。</p></div>
<h2>第一次读源码只跟这四处</h2>
<p>{c.ref(E+'llm_engine.py','LLMEngine.generate')} → {c.ref(E+'llm_engine.py','LLMEngine.add_request')} → {c.ref(E+'llm_engine.py','LLMEngine.step')} → {c.ref(E+'scheduler.py','Scheduler.postprocess')}</p>
{checkpoint('为什么总 token 数会比 num_cached_tokens 多 1？','采样结果已经追加进 token_ids，但该 token 还没作为下一轮输入计算 K/V。请求完成后 deallocate() 会把缓存计数清零，所以要分清“更新前”与“释放后”。')}
'''))
    pages.append(chapter('02-scheduler.html', '调度与分块 Prefill', '02 / 每轮应该计算谁', '阅读 scheduler.py。区分请求数量、token 预算和可用 KV 块这三个限制。', f'''
<h2>两个队列，一次选择</h2>
<p><code>waiting</code> 保存尚未完成 Prefill 的请求，也包括被抢占后重新排队的请求；<code>running</code> 保存准备继续 Decode 的请求。每轮先尝试安排 Prefill；如果安排到了，就直接返回这一批。只有没有 Prefill 被选中时，才进入 Decode 分支。</p>
{flow([('waiting','优先尝试安排 Prefill',''),('token 预算 / KV 块','检查是否能运行',''),('返回本轮 batch','Prefill 或 Decode','gpu')])}
<div class="note"><strong>这个版本不混合 Prefill 和 Decode。</strong><p>它实现了按迭代重组 batch 的基础连续批处理，但没有正式引擎中更复杂的混合调度。长 Prompt 连续占用 Prefill 轮次时，已有 Decode 请求可能等待。</p></div>
<h2>三个限制分别管什么</h2>
{table(['限制','代码中的含义'], [('max_num_seqs','本轮最多选多少个请求。'),('max_num_batched_tokens','Prefill 分支里累计安排的 token 预算；Decode 分支在本实现中主要按请求数与 KV 空间检查。'),('BlockManager.can_allocate / can_append','是否有足够缓存块容纳请求，或让请求继续增长。')])}
<h2>600 token 的 Prompt 怎么拆</h2>
<p>教学配置：每轮 token 预算为 256，A 的 Prompt 长 600，B 长 128，按 A、B 的顺序排队，缓存足够。两者暂不考虑前缀命中。</p>
{table(['轮次','本轮 Prefill','剩余工作'], [('1','A 的前 256 个 token','A 还剩 344；B 等待。'),('2','A 的中间 256 个 token','A 还剩 88；B 等待。'),('3','A 的最后 88 + B 的 128，共 216','两个请求都完成 Prefill，并各采样首个输出。'),('4','进入 Decode，各处理上轮生成的 1 个 token','两者都尚未结束时，一起继续生成。')])}
<p>对应 <code>min(num_tokens, remaining)</code>。当前实现只允许本轮第一个 Prefill 请求被切块；如果前面已经安排了请求，后一个请求放不下，就留到下轮。<strong>切分的是计算量；首次进入调度时，仍会为当前整个 Prompt 分配所需缓存块。</strong></p>
<p>{c.ref(E+'scheduler.py','Scheduler.schedule')} · {c.ref(E+'block_manager.py','BlockManager.allocate')}</p>
<h2>没有可用 KV 块时</h2>
<p>Decode 想继续增长却拿不到新块时，调度器会从其他运行请求中选择请求抢占；必要时抢占当前请求。它保留 token 序列，释放块引用，重新放回 <code>waiting</code>。之后可能命中尚未覆盖的前缀缓存，其余部分重新计算。</p>
{c.snippet(E+'scheduler.py','Scheduler.preempt')}
<p>这里没有 CPU KV 换出，也没有自动扩大 GPU 池。极端配置下，如果单个请求本身就超过池容量，简化调度器可能无法继续；不能把抢占理解成总能解决容量不足。</p>
{checkpoint('分块 Prefill 为何不等于“每次只分配一个 KV 块”？','Scheduler 第一次处理请求就调用 allocate()，后者为 seq.num_blocks 个逻辑块建立映射。num_scheduled_tokens 控制这轮算多少，和已经分配多少缓存位置是两件事。')}
'''))
    pages.append(chapter('03-kv-cache.html', 'KV 分页与地址映射', '03 / KV 到底存在哪里', '阅读 allocate_kv_cache()、BlockManager 和 block_table。把一块显存与一个 Python 对象分清。', f'''
<h2>GPU 存数据，CPU 记块号</h2>
{table(['所在位置','数据结构','保存内容'], [('GPU','ModelRunner.kv_cache','每层的 K/V 浮点向量，推理启动时预分配。'),('CPU','Block / BlockManager','块编号、引用计数、哈希、token IDs、空闲队列。'),('CPU → GPU','Sequence.block_table','先在 CPU 管理，再转为 GPU Tensor，告诉 Attention 去哪里读。')])}
{code('kv_cache.shape = [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]', '缓存池的维度（单 GPU；每层拿到自己的视图）')}
<p><code>2</code> 分别代表 K 和 V。一个管理块号在每一层对应各自的一块 K 和一块 V。默认 <code>block_size=256</code>；多 GPU 时 <code>num_kv_heads</code> 是这张卡负责的 KV 头数。</p>
<p>{c.ref(E+'model_runner.py','ModelRunner.allocate_kv_cache')} · {c.ref('nanovllm/config.py','Config')}</p>
<h2>显存池一次申请，请求按块领取</h2>
<p>先加载权重和预热，再根据总显存、显存利用预算、当前占用与预热峰值估算可分配块数，最后调用 <code>torch.empty(...)</code>。请求到达时领块；生成跨越块边界时，再从原池里领块。</p>
{code('每块字节数 = 2 × 层数 × block_size × 每卡 KV 头数 × head_dim × 每个元素字节数\n池的字节数 = num_blocks × 每块字节数', '容量估算；“一个位置”包含多个头的向量，不是一个浮点数')}
<div class="takeaway"><strong>预分配池确实占显存。</strong><p>分页的价值是让多个请求共享同一个容量池，避免每个请求都按最大可能长度预留空间。池在 Tensor 地址空间中可以连续；每个请求领取的块号无须连续。</p></div>
<h2>试着改变请求长度和 token 位置</h2>
<p>以下是地址映射演示，示意块号为 <code>[7, 2, 19, 4, 11, 6, 0, 15]</code>，不是当前 GPU 的实际分配结果。长度表示需要容纳 KV 的 token 数；块大小使用本地默认值 256。</p>
<div class="lab" id="kv-lab"><div class="lab-title"><h3>逻辑位置 → 缓存位置</h3><span class="tag">默认 256 / 块</span></div><div class="lab-controls"><label>需要缓存的 token 数（1—2048）<input id="kv-length" type="number" min="1" max="2048" value="600"></label><label>查看位置 p（从 0 开始）<input id="kv-position" type="range" min="0" max="599" value="300"><output id="kv-position-value" for="kv-position">300</output></label></div><div class="metrics" id="kv-metrics"><div><strong>3</strong><span>占用块数</span></div><div><strong>168</strong><span>末块未使用位置</span></div><div><strong>768</strong><span>分配的 token 容量</span></div></div><div class="blocks" id="kv-blocks"><div class="cache-block"><strong>逻辑块 0 → 块 7</strong>256 / 256</div><div class="cache-block selected"><strong>逻辑块 1 → 块 2</strong>256 / 256</div><div class="cache-block"><strong>逻辑块 2 → 块 19</strong>88 / 256</div></div><div class="mapping-result" id="kv-result" aria-live="polite">p = 300 → 逻辑块 1 → 块 2，偏移 44 → slot = 556</div><noscript><p>当前显示 600 token 的静态例子。开启 JavaScript 可拖动查看其他位置。</p></noscript></div>
{code('logical_block = p // block_size\noffset = p % block_size\nphysical_block = block_table[logical_block]\nslot = physical_block * block_size + offset', '地址公式；slot 是 token 槽编号，不是字节地址')}
<p>{c.ref(E+'model_runner.py','ModelRunner.prepare_decode')} · {c.ref(E+'sequence.py','Sequence.num_blocks')}</p>
<h2>为什么 may_append() 判断余数等于 1</h2>
<p>假设已经算好了 256 个 token 的 KV，然后采样出了第 257 个 token。此时 <code>len(seq)=257</code>，旧块已满；下一轮要处理这个新 token，因此在 <code>257 % 256 == 1</code> 时申请下一块。判断的是<strong>已经包含上轮采样结果</strong>的序列长度。</p>
{c.snippet(E+'block_manager.py','BlockManager.may_append')}
{checkpoint('600 token 需要 3 块，剩余 168 个位置能给另一个请求吗？','当前实现按整块分配。末块仍归这个请求，其他请求不能领取其中的 168 个位置。这是末块内部未使用空间；它与池中完全空闲的块不同。')}
'''))
    pages.append(chapter('04-prefix.html', '前缀复用与回收', '04 / 两个请求如何共用 KV', '阅读哈希索引、引用计数和空闲队列。复用的单位是已经计算好的完整前缀块。', f'''
<h2>先有相同上下文，才可能有相同 KV</h2>
<p>同样一段 token 放在不同前文之后，其 K/V 可以不同。nano 的块哈希把前面的哈希也纳入计算：</p>
{code('h0 = hash(第 0 块 token)\nh1 = hash(h0, 第 1 块 token)\nh2 = hash(h1, 第 2 块 token)', '链式前缀哈希；实际使用 xxhash.xxh64()')}
<p><code>hash_to_block_id</code> 保存哈希到缓存块号的映射。查找时还比较当前块的 token IDs，并从前往后连续匹配，一处失败便停止。GPU 计算完成后，<code>hash_blocks()</code> 把新完成的完整块加入索引。</p>
<p>{c.ref(E+'block_manager.py','BlockManager.compute_hash')} · {c.ref(E+'block_manager.py','BlockManager.can_allocate')} · {c.ref(E+'block_manager.py','BlockManager.hash_blocks')}</p>
<h2>一个具体例子</h2>
<p>A 和 B 的 Prompt 都长 600，前 512 个 token 完全一致。A 已完成 Prefill，它的块表是示意值 <code>[7, 2, 19]</code>；B 到来后，可以得到 <code>[7, 2, 4]</code>。</p>
{table(['逻辑位置','请求 A','请求 B','处理方式'], [('前 256 个 token','块 7','块 7','共用，引用数从 1 变 2。'),('接下来 256 个 token','块 2','块 2','共用，引用数从 1 变 2。'),('最后 88 个 token','块 19','块 4','分别持有自己的尾块。')])}
<p>B 的 <code>num_cached_tokens=512</code>，只计算剩余 88 个输入 token 的新 Q/K/V。但这些新 Q 做 Attention 时，仍然要读取前面 512 个 token 的 KV。命中前缀减少重复 Prefill，不代表后续 Attention 可以忽略前文。</p>
<div class="note"><strong>本地版本会排除请求最后一个逻辑块的直接复用。</strong><p><code>can_allocate()</code> 遍历 <code>range(seq.num_blocks - 1)</code>。即使最后一块恰好填满，也留给重新计算，从而保留产生当前输出 logits 的执行路径。例如同样的 512-token Prompt，最多直接复用前 256 个，而不是直接跳过全部 512 个。</p></div>
<h2>“空闲”为什么还能命中</h2>
{table(['事件','引用数','块的状态'], [('A、B 共用','2','不能回收覆盖。'),('A 结束','1','B 仍在用。'),('B 结束','0','进入空闲队列；原 KV 与哈希可暂留。'),('新请求命中旧前缀','0 → 1','从空闲队列取回该块，直接复用。'),('空闲块被分给其他内容','0 → 1','删除旧哈希关联、重置元信息，随后写入新 KV。')])}
<p>这里没有按时间过期的 TTL。回收由引用计数和池空间压力驱动；<code>deallocate()</code> 不会释放整个 GPU Tensor。不要把空闲队列误认为“已经清零的显存”。</p>
<p>{c.ref(E+'block_manager.py','BlockManager.allocate')} · {c.ref(E+'block_manager.py','BlockManager.deallocate')} · {c.ref(E+'block_manager.py','BlockManager._allocate_block')}</p>
{checkpoint('能把还没有填满的尾块共享给两个继续增长的请求吗？','当前 nano 的前缀复用避开尾块，每个请求拥有自己的可写尾部，因此无需为这种场景实现完整的写时复制逻辑。共享的是已完成、无需继续修改的前缀块。')}
'''))
    pages.append(chapter('05-attention.html', '从块表到 Attention', '05 / GPU 如何读写这些块', '阅读 prepare_prefill()、prepare_decode() 和 Attention.forward()。把索引转换成一次真实计算。', f'''
<h2>模型先为本轮输入计算 Q/K/V</h2>
{flow([('输入 token','Embedding + 模型层','gpu'),('QKV 线性层','Q/K 归一化与 RoPE','gpu'),('写 K/V','按 slot_mapping 放进池','gpu'),('Attention','Q 读取上下文 K/V','gpu')])}
<p>Qwen3 的每一层都会生成自己的 Q/K/V。K 经过旋转位置编码后写入缓存；V 也写入缓存；Q 只参与当前计算，不作为长期 KV 缓存保存。</p>
<p>{c.ref('nanovllm/models/qwen3.py','Qwen3Attention.forward')}</p>
<h2>两张映射，职责不同</h2>
{table(['输入','谁准备','GPU 用它做什么'], [('slot_mapping','ModelRunner','确定每个新 token 的 K/V 应该写入哪个槽位。'),('block_tables','ModelRunner','确定当前请求已有 KV 分布在哪些块，供 Attention 读取。'),('context_lens','prepare_decode()','限制读取到当前请求的有效 token 数，不读取末块空位。'),('cu_seqlens_q / cu_seqlens_k','prepare_prefill()','在拼接的多请求输入中，标记各请求 Q/K 的边界。')])}
<p>这些元信息先由 CPU 整理，再转换成 GPU Tensor。缓存池始终在 GPU 上，并不因为每轮调用 Python 就来回复制整份 KV。</p>
<h2>写入：一个 token 对应一个槽</h2>
{code('slot = slot_mapping[token_index]\nD = num_kv_heads * head_dim\nk_cache_flat[slot * D : (slot + 1) * D] = 当前 token 的 K\nv_cache_flat[slot * D : (slot + 1) * D] = 当前 token 的 V', '按单层解释 Triton store_kvcache_kernel 的索引，不是可执行 Python')}
<p>真实代码使用 Triton 并行写入，<code>slot=-1</code> 表示跳过，这也用于 CUDA Graph 的填充位置。</p>
<p>{c.ref(L+'attention.py','store_kvcache_kernel')} · {c.ref(L+'attention.py','store_kvcache')}</p>
<h2>读取：根据阶段选择内核</h2>
{table(['阶段','本轮输入','Attention 路径'], [('没有旧缓存的 Prefill','整个新输入或第一块输入','flash_attn_varlen_func：直接使用当前算出的 K/V，同时把 K/V 写入池。'),('前缀命中 / 后续 Prefill 分块','尚未计算的 token','flash_attn_varlen_func：通过 block_table 读取池中的旧 KV 与本轮新 KV。'),('Decode','每个请求上轮采样出的一个 token','flash_attn_with_kvcache：读取分页 KV，加上有效上下文长度。')])}
<p>{c.ref(L+'attention.py','Attention.forward')} · {c.ref(E+'model_runner.py','ModelRunner.prepare_prefill')} · {c.ref(E+'model_runner.py','ModelRunner.prepare_decode')}</p>
<details class="checkpoint"><summary>继续看：Prefill 中 Q 和 K 的长度为什么不同</summary><div><p>假设已有 512 个缓存 token，这轮计算 88 个新 token：Q 的长度是 88，K/V 的有效长度是 600。因为每个新 token 需要关注已有上下文。ModelRunner 用不同的 cu_seqlens_q/k 描述这两种长度，并把块表交给 Attention。</p></div></details>
<div class="takeaway"><strong>分页管理与 FlashAttention 可以一起使用。</strong><p>分页机制决定 KV 的存放和寻址方式；FlashAttention 通过分块计算与在线 softmax，减少注意力计算中的中间数据读写。nano 负责缓存组织和调用，完整 Attention 内核来自 flash-attn 库。</p></div>
{checkpoint('命中 512-token 前缀后，GPU 完全不读取这 512 个 token 吗？','仍会读取它们的 KV，供新的 Q 做 Attention。省掉的是对这段前缀重新运行模型得到 K/V 的工作。')}
'''))
    pages.append(chapter('06-execution.html', 'CUDA Graph 与编译', '06 / 减少重复发起计算的开销', '这章可以第二遍再读。区分图重放、局部编译和算法本身的计算量。', f'''
<h2>为什么 GPU 很快，CPU 仍可能忙</h2>
<p>一次 Decode 包含很多张量操作。如果每轮都由 Python 和运行时逐项准备与发起，CPU 的开销可能变得明显。CUDA Graph 让一段兼容的 GPU 操作序列先被捕获，再反复重放。</p>
{flow([('预热和捕获','为若干 batch 大小建立图',''),('填入本轮数据','token、位置、块表、长度',''),('graph.replay()','执行已捕获的模型流程','gpu'),('logits + 采样','得到本轮输出 token','gpu')])}
<p>nano 捕获的是 <code>self.model(...)</code> 这段模型主干；<code>compute_logits()</code> 和采样在该捕获范围之外。运行时从已捕获 batch 大小中选择能装下当前请求数的一档，剩余槽位用长度 0 和无效写入位置处理。</p>
<p>{c.ref(E+'model_runner.py','ModelRunner.capture_cudagraph')} · {c.ref(E+'model_runner.py','ModelRunner.run_model')}</p>
<h2>它不会把算力成本变成零</h2>
<p>矩阵乘法、Attention、读取权重和 KV 仍然发生。图重放主要减少重复组织和发起操作的成本。能节省多少，取决于实际负载里 CPU 发起开销占多少。</p>
<h2>torch.compile 处理局部计算</h2>
<p>本地代码给 RMSNorm、残差相加与归一化、RoPE、SiLU 与乘法、采样等函数加了 <code>@torch.compile</code>。编译器可以优化这些函数中的张量计算，具体生成的内核取决于运行环境。</p>
{c.snippet(L+'activation.py','SiluAndMul.forward')}
<p>{c.ref(L+'layernorm.py','RMSNorm.add_rms_forward')} · {c.ref(L+'rotary_embedding.py','RotaryEmbedding.forward')} · {c.ref(L+'sampler.py','Sampler.forward')}</p>
<div class="note"><strong>第一次调试建议 enforce_eager=True。</strong><p>这会绕过这里的 CUDA Graph 路径，更容易按普通调用链理解执行；它不会自动移除其他函数上的 <code>@torch.compile</code>。这一点与“完全关闭所有编译”不同。</p></div>
<h2>采样只看最小功能</h2>
<p>Sampler 将 logits 除以温度，做 softmax 后随机采样。当前 SamplingParams 只有 temperature、max_tokens、ignore_eos；代码明确要求温度大于 1e-10，不支持通过 temperature=0 切换贪心采样。</p>
<p>{c.ref('nanovllm/sampling_params.py','SamplingParams')} · {c.ref(L+'sampler.py','Sampler.forward')}</p>
{checkpoint('CUDA Graph 重放的是上一次算出来的答案吗？','不是。每轮先把新 token、位置和块表等数据填入固定的缓冲区，再执行同一段 GPU 操作流程。重用的是操作组织方式，计算结果仍取决于本轮输入。')}
'''))
    pages.append(chapter('07-parallel.html', '多 GPU 与进程通信', '07 / 两张 GPU 怎么一起推理', '先看一台机器上的张量并行。把 CPU 发任务和 GPU 交换张量分开理解。', f'''
<h2>单 GPU 先记住这一点</h2>
<p>调用进程内同时有 Scheduler 和 rank 0 的 ModelRunner。配置 TP&gt;1 时，引擎通过 <code>multiprocessing</code> 的 spawn 方式，为 rank 1、rank 2 等启动额外进程；每个进程设置自己的 CUDA 设备。</p>
{table(['位置','职责'], [('主 CPU 进程','维护请求与调度器；持有 rank 0 ModelRunner；控制 GPU 0。'),('其他 CPU 工作进程','持有各自 ModelRunner；接收执行命令；控制对应 GPU。'),('每张 GPU','保存本卡模型分片与本卡 KV 头的缓存；完成局部张量运算。')])}
<p>{c.ref(E+'llm_engine.py','LLMEngine.__init__')} · {c.ref(E+'model_runner.py','ModelRunner.__init__')}</p>
<h2>CPU 发任务：共享内存 + Event</h2>
<p>rank 0 把方法名和参数用 pickle 序列化，写入一块 1 MiB 的 CPU 共享内存，然后设置各 worker 的 Event。worker 等待 Event，读取并反序列化，调用相应方法。这里传递的是执行命令与请求元信息，不是整份 GPU KV。</p>
{flow([('rank 0','序列化 run 与请求参数',''),('CPU 共享内存','写消息；Event 通知',''),('rank 1…','读消息，调用 ModelRunner',''),('各自 GPU','运行本卡模型分片','gpu')])}
<p>这个本地路径没有使用 ZMQ。Decode 序列化还会减少请求内容，只发送下一轮需要的最后一个 token 等状态，而不总是复制完整历史。</p>
<p>{c.ref(E+'model_runner.py','ModelRunner.write_shm')} · {c.ref(E+'model_runner.py','ModelRunner.read_shm')} · {c.ref(E+'sequence.py','Sequence.__getstate__')}</p>
<h2>GPU 协作：分片权重 + NCCL</h2>
<p>以数学记号 <code>Y=XW</code> 解释，按输出维度拆 W，可以让两张卡得到 Y 的不同部分；按输入维度拆 W，则两张卡产生同一输出的部分和，最后相加。</p>
{code('按输出维度拆：W = [W0 | W1]      → Y0 = XW0，Y1 = XW1\n按输入维度拆：X = [X0 | X1]      → Y = X0W0 + X1W1', '数学示意；PyTorch F.linear 的权重存储形状是 [out_features, in_features]')}
<p><code>ColumnParallelLinear</code> 切分输出特征；<code>RowParallelLinear</code> 切分输入特征并用 <code>dist.all_reduce()</code> 汇总部分结果。Q/K/V 按头拆分；词表嵌入和输出头也有分片处理。最终 logits 收集到 rank 0，由它采样输出。</p>
<p>{c.ref(L+'linear.py','ColumnParallelLinear')} · {c.ref(L+'linear.py','RowParallelLinear.forward')} · {c.ref(L+'embed_head.py','ParallelLMHead.forward')}</p>
<div class="note"><strong>这个实现面向单机 TP。</strong><p>初始化使用 localhost 的 NCCL rendezvous 地址，本机共享内存传命令；没有完整的跨节点部署、流水线并行或专家并行调度。配置还要求相关头数、词表等维度满足切分条件。</p></div>
{checkpoint('共享内存队列负责传递 all-reduce 的张量吗？','不负责。共享内存 + Event 是 CPU 进程之间的命令通道；模型执行中的张量集体通信由 torch.distributed 的 NCCL 后端完成。')}
'''))
    pages.append(chapter('08-practice.html', '动手实验与验收', '08 / 用小实验验证理解', '先在 CPU 上验证真实缓存与调度代码；有 NVIDIA GPU 后，再运行完整模型。', f'''
<h2>Mac 上也能做的三个实验</h2>
<p>随文档附带 <code>cpu_lab.py</code>。它直接加载本仓库的 Sequence、BlockManager 和 Scheduler，通过独立的包加载方式避开 GPU 引擎入口；配置使用简单对象。模型计算由人为指定的 token 结果代替，<strong>只验证 CPU 状态与调度，不计算 K/V 向量，也不测推理性能</strong>。</p>
{code('cd ~/Desktop/source_code/nano-vllm\n.venv/bin/python docs/web/cpu_lab.py', '使用项目现有虚拟环境；需要 numpy 和 xxhash，无须 GPU 或模型权重')}
{table(['实验','观察什么','预期结果'], [('1. 分配、共享、释放','A/B 各 600 token，前 512 相同','各需 3 块；B 直接复用 A 的前两块；共享引用数变 2；结束后回到空闲队列。'),('2. 新 token 跨块','已缓存 256，然后采样出第 257 个','下一轮分配第二块，映射到块内偏移 0。'),('3. 分块调度','A=600，B=128，每轮预算 256','三轮 Prefill 为 256 / 256 / 216；之后一轮 Decode 同时处理两个请求。')])}
<p>实验还检查“最后一个满块不直接复用”的边界。打印的块号来自该次 CPU 管理器的真实结果，和上一章任意选择的图解块号可能不同。</p>
<h2>有 NVIDIA GPU 后跑一次最小生成</h2>
<p>在可用 CUDA、NCCL、Triton 与 FlashAttention 的环境中，先准备本地 Qwen3-0.6B 权重。项目的 <code>example.py</code> 默认读取 <code>~/huggingface/Qwen3-0.6B/</code>，应用 chat template 后调用 <code>generate()</code>。</p>
{code('python example.py', '在已配置好依赖与模型权重的 nano-vllm 项目根目录运行')}
<p>第一次用 <code>tensor_parallel_size=1</code> 和 <code>enforce_eager=True</code>。先确认能生成，再逐项调整；不要同时切换模型、TP 和图模式。</p>
<p>{c.ref('example.py','main')}</p>
<h2>加观察点时，记录这几个值</h2>
{code('seq_id, status, num_tokens, num_cached_tokens,\nnum_scheduled_tokens, block_table, len(free_block_ids)', '分别在 schedule() 后和 postprocess() 后观察，区分两个时间点')}
<p>对比 <code>num_tokens</code> 与 <code>num_cached_tokens</code>，找到“采样已完成、KV 尚未计算”的时刻；对比引用计数与空闲队列，找到“请求结束、池依然占用显存”的原因。</p>
<h2>读完能回答这五个问题就够了</h2>
{checkpoint('为什么 Prefill 可以一次处理很多 token，Decode 通常每个请求只处理一个？','Prompt token 已全部给定，可以使用因果掩码并行计算；普通自回归 Decode 要等当前预测结果出来，才能确定下一个输入。这里不包含投机解码。')}
{checkpoint('块表为什么能让请求的 KV 不连续？','逻辑顺序由请求的 block_table 决定。Attention 用逻辑块索引查实际块号，再加块内偏移；不要求相邻逻辑块在池里相邻。')}
{checkpoint('Prefill 分块是否自动减少已经预留的 Prompt KV 容量？','这个版本没有按 Prefill 切块逐块预留整个 Prompt；首次 allocate() 就为当前 Prompt 的全部逻辑块建立映射。分块主要限制每轮执行的 token 数。')}
{checkpoint('公共前缀会不会每次复制一份 KV？','命中后，块表直接引用同一缓存块并增加引用计数。其余未命中部分单独分配。')}
{checkpoint('Python 调度变快，是否必然同等比例加速推理？','不会。整体收益取决于 CPU 调度是否限制 GPU 工作，以及计算、显存读写、通信各占多少时间。先定位瓶颈，再判断优化对象。')}
'''))
    pages.append(chapter('09-reference.html', '术语与功能地图', '09 / 随用随查的源码地图', '忘记一个名词时回来查。先读主线，不必一次记住整张表。', f'''
<h2>功能 → 源码</h2>
{table(['功能','本地入口','范围'], [('请求循环',c.ref(E+'llm_engine.py','LLMEngine.step'),'同步离线生成，迭代推进请求。'),('连续批处理 / 分块 Prefill',c.ref(E+'scheduler.py','Scheduler.schedule'),'Prefill 优先，当前不混合 Prefill 与 Decode。'),('分页 KV / 前缀复用',c.ref(E+'block_manager.py','BlockManager'),'块池、链式哈希、引用计数。'),('分页 Attention',c.ref(L+'attention.py','Attention.forward'),'调用 flash-attn 的缓存与块表接口。'),('CUDA Graph',c.ref(E+'model_runner.py','ModelRunner.capture_cudagraph'),'捕获模型主干，主要用于 Decode。'),('局部编译',c.ref(L+'layernorm.py','RMSNorm'),'RMSNorm、RoPE、激活函数、采样等。'),('张量并行',c.ref(L+'linear.py','RowParallelLinear'),'单机多 GPU 分片与集体通信。'),('CPU 进程通信',c.ref(E+'model_runner.py','ModelRunner.write_shm'),'共享内存 + Event + pickle。'),('模型结构',c.ref('nanovllm/models/qwen3.py','Qwen3ForCausalLM'),'当前执行器直接使用 Qwen3 路径。'),('采样',c.ref('nanovllm/sampling_params.py','SamplingParams'),'温度、生成长度、EOS；没有完整 top-k/top-p 参数。')])}
<h2>容易混淆的名词</h2>
<dl class="glossary"><dt>Token</dt><dd>分词后的离散 ID。token ID 与它的浮点 K/V 向量是不同数据。</dd><dt>Prefill / Decode</dt><dd>Prefill 处理 Prompt 或需要重算的序列；Decode 处理每个请求上轮刚采样出的 token，并产生下一个 token。</dd><dt>Block / Page</dt><dd>此处指 KV 管理的固定大小分配单元，不等同于操作系统的页。本地默认一个块对应 256 个 token 位置。</dd><dt>Block table</dt><dd>请求的逻辑块到缓存池块号的映射。它很小；实际 K/V 数据仍在 GPU 缓存池。</dd><dt>Slot mapping</dt><dd>为本轮新 token 指定写入槽位。读取历史则依赖块表和有效长度。</dd><dt>Prefix caching</dt><dd>匹配完整前缀块，复用已经算好的 KV，省去重复 Prefill。</dd><dt>FlashAttention</dt><dd>通过分块和在线 softmax 减少中间注意力矩阵的显存读写。nano 调用现成内核。</dd><dt>Tensor parallelism</dt><dd>把同一层的权重/特征拆到多张 GPU 协作。不是把前几层放一张卡、后几层放另一张卡的流水线并行。</dd><dt>CUDA Graph</dt><dd>捕获并重放 GPU 操作流程，减少重复发起的开销。不是缓存模型答案。</dd></dl>
<h2>这份项目之外，下一步学什么</h2>
<p>当前核心路径没有集成投机解码、完整量化适配、LoRA、多模态、在线流式 API、跨节点编排和完善的生产监控。学完后可去正式 vLLM 阅读服务入口与 EngineCore，去 SGLang 对比 Radix Cache；迁移的是机制理解，不是逐个类名的一一对应。</p>
<h2>文档如何更新</h2>
{code('python3 docs/web/build.py\npython3 docs/web/validate.py', '项目根目录；重新生成源码快照并检查本地链接、行号与哈希')}
<p>重新生成会更新快照和引用行号，<strong>不会自动改写解释文字</strong>。如果上游算法发生变化，需要一并复核相应章节。阅读入口始终是 <code>docs/web/index.html</code>，所有页面都可以直接离线打开。</p>
'''))
    return pages
