"""Qwen3 模型组装，去掉 nn.Module 包装。

结构和 nanovllm/models/qwen3.py 一一对应，但每一级都拆成了「数据容器 + 自由
函数」：容器用 __slots__ 存裸 Tensor，前向是模块级函数。

权重不重新加载，而是用 adopt() 从一个已经 load_model 过的 nn.Module 模型里把
Tensor 直接接管过来（共享同一块显存，不复制）。这样两套实现在数值上必然等价，
差异只剩下调用路径本身。
"""
import torch

from nanovllm.models.qwen3 import Qwen3ForCausalLM

from layers_without_torch.activation import silu_and_mul
from layers_without_torch.attention import AttentionState, attention
from layers_without_torch.embed_head import VocabWeight, embedding, lm_head
from layers_without_torch.layernorm import RMSNormWeight, rms_norm
from layers_without_torch.linear import LinearWeight, linear, row_linear
from layers_without_torch.rotary_embedding import RopeCache, apply_rope


class Qwen3AttentionWeight:
    __slots__ = ("qkv_proj", "o_proj", "q_norm", "k_norm", "rope", "attn",
                 "q_size", "kv_size", "num_heads", "num_kv_heads", "head_dim", "qkv_bias")


class Qwen3MLPWeight:
    __slots__ = ("gate_up_proj", "down_proj")


class Qwen3DecoderLayerWeight:
    __slots__ = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")


class Qwen3ModelWeight:
    __slots__ = ("embed_tokens", "layers", "norm")


class Qwen3CausalLMWeight:
    __slots__ = ("model", "lm_head")


def qwen3_attention(w: Qwen3AttentionWeight, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    qkv = linear(w.qkv_proj, hidden_states)
    q, k, v = qkv.split([w.q_size, w.kv_size, w.kv_size], dim=-1)
    q = q.view(-1, w.num_heads, w.head_dim)
    k = k.view(-1, w.num_kv_heads, w.head_dim)
    v = v.view(-1, w.num_kv_heads, w.head_dim)
    if not w.qkv_bias:
        q = rms_norm(w.q_norm, q)
        k = rms_norm(w.k_norm, k)
    q, k = apply_rope(w.rope, positions, q, k)
    o = attention(w.attn, q, k, v)
    return row_linear(w.o_proj, o.flatten(1, -1))


def qwen3_mlp(w: Qwen3MLPWeight, x: torch.Tensor) -> torch.Tensor:
    gate_up = linear(w.gate_up_proj, x)
    x = silu_and_mul(gate_up)
    return row_linear(w.down_proj, x)


def qwen3_decoder_layer(w: Qwen3DecoderLayerWeight, positions: torch.Tensor,
                        hidden_states: torch.Tensor, residual: torch.Tensor | None):
    if residual is None:
        hidden_states, residual = rms_norm(w.input_layernorm, hidden_states), hidden_states
    else:
        hidden_states, residual = rms_norm(w.input_layernorm, hidden_states, residual)
    hidden_states = qwen3_attention(w.self_attn, positions, hidden_states)
    hidden_states, residual = rms_norm(w.post_attention_layernorm, hidden_states, residual)
    hidden_states = qwen3_mlp(w.mlp, hidden_states)
    return hidden_states, residual


def qwen3_model(w: Qwen3ModelWeight, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    hidden_states = embedding(w.embed_tokens, input_ids)
    residual = None
    # layers 是 tuple 而不是 nn.ModuleList：迭代不经过 _modules 字典
    for layer in w.layers:
        hidden_states, residual = qwen3_decoder_layer(layer, positions, hidden_states, residual)
    hidden_states, _ = rms_norm(w.norm, hidden_states, residual)
    return hidden_states


def qwen3_forward(w: Qwen3CausalLMWeight, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return qwen3_model(w.model, input_ids, positions)


def qwen3_compute_logits(w: Qwen3CausalLMWeight, hidden_states: torch.Tensor) -> torch.Tensor:
    return lm_head(w.lm_head, hidden_states)


def _adopt_linear(module) -> LinearWeight:
    return LinearWeight(module.weight.data, None if module.bias is None else module.bias.data,
                        module.tp_rank, module.tp_size)


def _adopt_rmsnorm(module) -> RMSNormWeight:
    return RMSNormWeight(module.weight.data, module.eps)


def _adopt_vocab(module) -> VocabWeight:
    return VocabWeight(module.weight.data, module.tp_rank, module.tp_size,
                       module.vocab_start_idx, module.vocab_end_idx)


def adopt(model: Qwen3ForCausalLM) -> Qwen3CausalLMWeight:
    """接管一个已加载权重的 nn.Module 模型，返回等价的无模块版本。

    Tensor 是共享的，不做拷贝。KV 池视图按调用时的状态读取；若在 adopt 之后才
    分配 KV 池，需要再调一次 bind_kv_cache。
    """
    # 所有层共用一份 cos/sin 表，对应原版 get_rope 的 lru_cache
    rope = RopeCache(model.model.layers[0].self_attn.rotary_emb.cos_sin_cache)

    layers = []
    for src in model.model.layers:
        src_attn = src.self_attn
        attn_w = Qwen3AttentionWeight()
        attn_w.qkv_proj = _adopt_linear(src_attn.qkv_proj)
        attn_w.o_proj = _adopt_linear(src_attn.o_proj)
        attn_w.qkv_bias = src_attn.qkv_bias
        attn_w.q_norm = None if src_attn.qkv_bias else _adopt_rmsnorm(src_attn.q_norm)
        attn_w.k_norm = None if src_attn.qkv_bias else _adopt_rmsnorm(src_attn.k_norm)
        attn_w.rope = rope
        attn_w.q_size = src_attn.q_size
        attn_w.kv_size = src_attn.kv_size
        attn_w.num_heads = src_attn.num_heads
        attn_w.num_kv_heads = src_attn.num_kv_heads
        attn_w.head_dim = src_attn.head_dim

        state = AttentionState(src_attn.attn.num_heads, src_attn.attn.head_dim,
                               src_attn.attn.scale, src_attn.attn.num_kv_heads)
        state.k_cache = src_attn.attn.k_cache
        state.v_cache = src_attn.attn.v_cache
        attn_w.attn = state

        mlp_w = Qwen3MLPWeight()
        mlp_w.gate_up_proj = _adopt_linear(src.mlp.gate_up_proj)
        mlp_w.down_proj = _adopt_linear(src.mlp.down_proj)

        layer_w = Qwen3DecoderLayerWeight()
        layer_w.self_attn = attn_w
        layer_w.mlp = mlp_w
        layer_w.input_layernorm = _adopt_rmsnorm(src.input_layernorm)
        layer_w.post_attention_layernorm = _adopt_rmsnorm(src.post_attention_layernorm)
        layers.append(layer_w)

    model_w = Qwen3ModelWeight()
    model_w.embed_tokens = _adopt_vocab(model.model.embed_tokens)
    model_w.layers = tuple(layers)
    model_w.norm = _adopt_rmsnorm(model.model.norm)

    causal_w = Qwen3CausalLMWeight()
    causal_w.model = model_w
    causal_w.lm_head = _adopt_vocab(model.lm_head)
    return causal_w


def bind_kv_cache(w: Qwen3CausalLMWeight, kv_cache: torch.Tensor):
    """把 KV 池按层切成视图挂上去，对应 ModelRunner.allocate_kv_cache 的后半段。"""
    for layer_id, layer in enumerate(w.model.layers):
        layer.self_attn.attn.k_cache = kv_cache[0, layer_id]
        layer.self_attn.attn.v_cache = kv_cache[1, layer_id]
