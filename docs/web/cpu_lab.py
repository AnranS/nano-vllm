#!/usr/bin/env python3
"""Exercise real CPU bookkeeping code, without importing or executing the GPU engine.

Requires numpy + xxhash. No model weights, torch, transformers, or CUDA are loaded.
Config is a SimpleNamespace and sampled tokens are synthetic. This is not inference.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def load_cpu_classes(repo):
    # This executable uses a separate process. Supply package paths without
    # running nanovllm/__init__.py, which otherwise imports the GPU engine.
    if 'nanovllm' in sys.modules:
        raise RuntimeError('Run cpu_lab.py as a standalone process.')
    for name, directory in [('nanovllm', repo / 'nanovllm'), ('nanovllm.engine', repo / 'nanovllm/engine')]:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    config_module = ModuleType('nanovllm.config')
    config_module.Config = SimpleNamespace
    sys.modules['nanovllm.config'] = config_module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, repo / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    params = load('nanovllm.sampling_params', 'nanovllm/sampling_params.py')
    seq = load('nanovllm.engine.sequence', 'nanovllm/engine/sequence.py')
    blocks = load('nanovllm.engine.block_manager', 'nanovllm/engine/block_manager.py')
    sched = load('nanovllm.engine.scheduler', 'nanovllm/engine/scheduler.py')
    seq.Sequence.block_size = 256
    return params.SamplingParams, seq.Sequence, blocks.BlockManager, sched.Scheduler


def main(repo):
    SamplingParams, Sequence, BlockManager, Scheduler = load_cpu_classes(repo)
    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    print('CPU 教学实验：运行真实管理器；没有 GPU KV 数据，也没有模型推理。')

    manager = BlockManager(32, 256)
    a = Sequence(list(range(600)), params)
    assert manager.can_allocate(a) == 0
    manager.allocate(a, 0)
    assert len(a.block_table) == 3
    # Mark the synthetic prefill as complete at the bookkeeping level only.
    a.num_scheduled_tokens = 600
    manager.hash_blocks(a)
    a.num_cached_tokens = 600
    a.num_scheduled_tokens = 0
    b = Sequence(list(range(512)) + list(range(2000, 2088)), params)
    cached = manager.can_allocate(b)
    assert cached == 2
    manager.allocate(b, cached)
    assert b.num_cached_tokens == 512
    assert a.block_table[:2] == b.block_table[:2]
    assert a.block_table[2] != b.block_table[2]
    shared = a.block_table[:2]
    assert all(manager.blocks[i].ref_count == 2 for i in shared)
    print(f'实验 1：A={a.block_table}，B={b.block_table}；B 命中 {b.num_cached_tokens} 个 token。')
    manager.deallocate(a)
    assert all(manager.blocks[i].ref_count == 1 for i in shared)
    manager.deallocate(b)
    assert len(manager.free_block_ids) == 32
    assert not manager.used_block_ids
    later = Sequence(list(range(600)), params)
    assert manager.can_allocate(later) == 2
    manager.allocate(later, 2)
    assert later.block_table[:2] == shared
    manager.deallocate(later)
    print('        两个请求释放后空闲块恢复为 32，旧前缀仍可命中。')

    boundary_manager = BlockManager(4, 256)
    boundary = Sequence(list(range(256)), params)
    boundary_manager.allocate(boundary, 0)
    boundary.num_scheduled_tokens = 256
    boundary_manager.hash_blocks(boundary)
    boundary.num_cached_tokens = 256
    boundary.num_scheduled_tokens = 0
    boundary.append_token(999)
    assert len(boundary) == 257
    assert boundary_manager.can_append(boundary)
    boundary_manager.may_append(boundary)
    assert len(boundary.block_table) == 2
    assert boundary.last_block_num_tokens == 1
    slot = boundary.block_table[-1] * 256 + boundary.last_block_num_tokens - 1
    assert slot % 256 == 0
    print(f'实验 2：第 257 个 token 的块表={boundary.block_table}，slot={slot}，块内偏移=0。')

    same_full_block = Sequence(list(range(256)), params)
    assert boundary_manager.can_allocate(same_full_block) == 0
    same_two_blocks = Sequence(list(range(512)), params)
    assert boundary_manager.can_allocate(same_two_blocks) == 1
    print('        边界检查：相同 256-token 请求不直接复用尾块；512-token 请求可复用前 256 个。')

    scheduler = Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=256,
                                         eos=-1, kvcache_block_size=256, num_kvcache_blocks=32))
    a = Sequence(list(range(600)), params)
    b = Sequence(list(range(3000, 3128)), params)
    scheduler.add(a)
    scheduler.add(b)
    labels = {a.seq_id: 'A', b.seq_id: 'B'}
    observed = []
    for step in range(4):
        seqs, prefill = scheduler.schedule()
        workloads = [(labels[s.seq_id], s.num_scheduled_tokens) for s in seqs]
        observed.append((prefill, workloads))
        print(f'实验 3 / 第 {step + 1} 轮：{"Prefill" if prefill else "Decode"} {workloads}')
        # Fake sampled IDs allow postprocess to advance; partial-prefill IDs
        # are ignored by the real scheduler until the prompt is finished.
        scheduler.postprocess(seqs, [10000 + step] * len(seqs), prefill)
    assert observed == [(True, [('A', 256)]), (True, [('A', 256)]),
                        (True, [('A', 88), ('B', 128)]), (False, [('A', 1), ('B', 1)])]
    assert scheduler.is_finished()
    assert len(scheduler.block_manager.free_block_ids) == 32
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    print('全部预期成立；没有加载 torch / triton，也没有运行 GPU。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    main(args.repo.resolve())
