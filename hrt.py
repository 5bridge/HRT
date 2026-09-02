#!/usr/bin/env python3

from __future__ import annotations

import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict

class ModelConfig:
    def __init__(
        self,
        d_model: int = 512,
        n_outer_latents: int = 128,
        n_outer_cycles: int = 3,
        n_outer_heads: int = 8,
        n_inner_cycles: int = 6,
        n_inner_heads: int = 8,
        n_broadcast_cycles: int = 0,
        n_latent_heads: int = 8,
        d_ff: int = 2048,
        local_conv_kernel: int = 7,
        dropout: float = 0.0,
        routing_k: int = 32,
        n_center_latents: int = 4,
        max_seq_len: int = 2048,
        use_grad_ckpt: bool = False,
        proj_chunk: int = 0,
        causal: bool = True,
        harp_max_heads: int = 8,
        harp_tl_scale: float = 1.0,
        use_internalization: bool = False,
        internalization_steps: int = 3,
        return_internalization_loss: bool = False,
        use_adaptive_halting: bool = False,
        adaptive_epsilon: float = 1e-6,
        max_adaptive_steps: int = 10,
        use_jfb: bool = True,
        use_qk_norm: bool = True,
        use_rezero: bool = True,
        rezero_init: float = 1e-4,
        use_compile: bool = False,
        compile_mode: str = "default",
        cycle_drop: bool = False,
        min_inner_cycles: int = 2,
        use_compaction: bool = True,
        d_compact: int | None = None,
        compaction_loss_weight: float = 0.1,
        vocab_size: int = 0,
        use_ib: bool = False,
        macro_jump: int = 4,
        btt_max_levels: int = 40,
        merger_path: str = "merger.pt",
        use_q_cache: bool = True,  
        mask_cache_mb: float = 256.0,  
    ):
        self.d_model = d_model
        self.n_outer_latents = n_outer_latents
        self.n_outer_cycles = n_outer_cycles
        self.n_outer_heads = n_outer_heads
        self.n_inner_cycles = n_inner_cycles
        self.n_inner_heads = n_inner_heads
        self.n_broadcast_cycles = n_broadcast_cycles
        self.n_latent_heads = n_latent_heads
        self.d_ff = d_ff
        self.local_conv_kernel = local_conv_kernel
        self.dropout = dropout
        self.routing_k = routing_k
        self.n_center_latents = n_center_latents
        self.max_seq_len = max_seq_len
        self.use_grad_ckpt = use_grad_ckpt
        self.proj_chunk = proj_chunk
        self.causal = causal
        self.harp_max_heads = harp_max_heads
        self.harp_tl_scale = harp_tl_scale
        self.use_internalization = use_internalization
        self.internalization_steps = internalization_steps
        self.return_internalization_loss = return_internalization_loss
        self.use_adaptive_halting = use_adaptive_halting
        self.adaptive_epsilon = adaptive_epsilon
        self.max_adaptive_steps = max_adaptive_steps
        self.use_jfb = use_jfb
        self.use_qk_norm = use_qk_norm
        self.use_rezero = use_rezero
        self.rezero_init = rezero_init
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.cycle_drop = cycle_drop
        self.min_inner_cycles = max(1, min_inner_cycles)
        self.use_compaction = use_compaction
        self.d_compact = d_compact
        self.compaction_loss_weight = compaction_loss_weight
        self.vocab_size = vocab_size
        self.use_ib = use_ib and (vocab_size > 0)
        self.macro_jump = macro_jump
        self.btt_max_levels = btt_max_levels
        self.merger_path = merger_path
        self.use_q_cache = use_q_cache
        self.mask_cache_mb = mask_cache_mb

def apply_qk_norm(Q, K, qk_scale):
    Q = F.normalize(Q, dim=-1)
    K = F.normalize(K, dim=-1)
    g = qk_scale.exp().view(1, -1, 1, 1)
    return Q * g, K

def make_qk_scale(n_heads, head_dim):
    init = math.log(math.sqrt(head_dim))
    return nn.Parameter(torch.full((n_heads,), init, dtype=torch.float32))

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_model, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.out(F.silu(self.gate(x)) * self.proj(x)))


class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.0, causal=True):
        super().__init__()
        self.k = kernel_size
        self.dilation = 3
        self.causal = causal
        self.norm = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=d_model, bias=False)
        self.dw_conv_dilated = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                                         dilation=self.dilation, groups=d_model, bias=False)
        self.pw_conv = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.blend = nn.Parameter(torch.tensor(0.5))
        self.drop = nn.Dropout(dropout)
        self._dilated_pad = (3 - 1) * self.dilation

    def _pad(self, x, total):
        if self.causal:
            return F.pad(x, (total, 0))
        left = total // 2
        right = total - left
        return F.pad(x, (left, right))

    def _causal_dwconv_seg(self, x, weight, dilation, segment_ids):
        B, D, T = x.shape
        Kk = weight.shape[-1]
        seg = segment_ids
        y = x.new_zeros(B, D, T)
        for j in range(Kk):
            shift = (Kk - 1 - j) * dilation
            if shift == 0:
                xj = x
                valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
            else:
                xj = F.pad(x, (shift, 0))[..., :T]
                seg_shift = F.pad(seg, (shift, 0), value=-1)[..., :T]
                valid = seg_shift == seg
            wj = weight[:, 0, j].view(1, D, 1)
            y = y + wj * xj * valid.unsqueeze(1).to(x.dtype)
        return y

    def forward(self, x, segment_ids=None):
        r = x
        x = self.norm(x).transpose(1, 2)
        if segment_ids is None:
            xp = self._pad(x, self.k - 1)
            y1 = F.gelu(self.dw_conv(xp))
            xd = self._pad(x, self._dilated_pad)
            y2 = F.gelu(self.dw_conv_dilated(xd))
        else:
            assert self.causal, "segment-aware conv requires causal=True"
            y1 = F.gelu(self._causal_dwconv_seg(x, self.dw_conv.weight, 1, segment_ids))
            y2 = F.gelu(self._causal_dwconv_seg(x, self.dw_conv_dilated.weight, self.dilation, segment_ids))
        b = torch.sigmoid(self.blend)
        y = self.pw_conv(b * y1 + (1 - b) * y2).transpose(1, 2)
        return r + self.drop(y)

    def forward_last(self, new_emb, emb_win, segment_id=None, emb_win_segment_ids=None):
        if segment_id is not None and emb_win_segment_ids is not None:
            if (emb_win_segment_ids != segment_id).any():
                raise ValueError(
                    "forward_last: segment boundary within the window — "
                    "An incremental path does not support segment intersections. "
                    "At the segment boundary, recreate the cache using _init_generation_cache."
                )
        window = torch.cat([emb_win, new_emb], dim=1)
        r = new_emb
        x = self.norm(window).transpose(1, 2)

        xp = F.pad(x, (self.k - 1, 0))
        y1 = F.gelu(self.dw_conv(xp))[:, :, -1:]

        xd = F.pad(x, (self._dilated_pad, 0))
        y2 = F.gelu(self.dw_conv_dilated(xd))[:, :, -1:]

        b = torch.sigmoid(self.blend)
        y = self.pw_conv(b * y1 + (1 - b) * y2).transpose(1, 2)
        return r + self.drop(y)


class HARP(nn.Module):
    """Hierarchical Adaptive Relational Positioning."""
    def __init__(self, n_heads_max, K, max_seq_len=8192, tl_scale=1.0):
        super().__init__()
        self.n_heads_max = n_heads_max
        self.K = K
        self.max_T = max_seq_len
        slopes = torch.tensor(
            [2.0 ** (-8.0 * i / n_heads_max) for i in range(1, n_heads_max + 1)],
            dtype=torch.float32,
        )
        self.log_slopes = nn.Parameter(slopes.log())
        self.ll_bias = nn.Parameter(torch.zeros(n_heads_max, K, K))
        nn.init.normal_(self.ll_bias, std=0.02)
        self.tl_scale = nn.Parameter(torch.tensor(float(tl_scale)))

    @property
    def slopes(self) -> torch.Tensor:
        return self.log_slopes.exp()

    def _budget_centers(self, T, device, K=None):
        K = K or self.K
        i = torch.arange(K, device=device, dtype=torch.float32)
        return (i + 0.5) * T / K

    def token_to_latent(self, T, H, device, dtype, K=None, base_T=None):
        center_T = base_T if base_T is not None else T
        centers = self._budget_centers(center_T, device, K)
        tok_pos = torch.arange(T, device=device, dtype=torch.float32)
        rel = (tok_pos[:, None] - centers[None, :]).abs()
        slopes = self.slopes[:H]
        bias = -slopes.view(H, 1, 1) * rel[None] * self.tl_scale
        return bias.unsqueeze(0).to(dtype)

    def token_to_latent_inc(self, prev_bias, T_new, H, device, dtype, K=None, base_T=None):
        K = K or self.K
        center_T = base_T if base_T is not None else T_new
        centers = self._budget_centers(center_T, device, K)
        slopes = self.slopes[:H]
        tok_pos = torch.tensor([T_new - 1], device=device, dtype=torch.float32)
        rel = (tok_pos[:, None] - centers[None, :]).abs()
        new_col = -slopes.view(1, H, 1, 1) * rel.view(1, 1, K, 1) * self.tl_scale
        new_col = new_col.to(dtype)
        return torch.cat([prev_bias.to(dtype), new_col], dim=-1)

    def latent_to_token(self, T, H, device, dtype, K=None, base_T=None):
        return self.token_to_latent(T, H, device, dtype, K, base_T).transpose(-1, -2)

    def latent_to_latent(self, H, device, dtype, K=None):
        K = K or self.K
        return self.ll_bias[:H, :K, :K].unsqueeze(0).to(device=device, dtype=dtype)

class LatentSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, use_qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.d = d_model // n_heads; self.drop = dropout
        self.use_qk_norm = use_qk_norm
        self.norm = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.res_scale = nn.Parameter(torch.ones(1))
        if use_qk_norm:
            self.qk_scale = make_qk_scale(n_heads, self.d)

    def forward(self, x, attn_mask=None):
        B, K, D = x.shape
        n = self.norm(x)
        Q = self.Wq(n).view(B, K, self.h, self.d).transpose(1, 2)
        Kk = self.Wk(n).view(B, K, self.h, self.d).transpose(1, 2)
        V = self.Wv(n).view(B, K, self.h, self.d).transpose(1, 2)
        if self.use_qk_norm:
            Q, Kk = apply_qk_norm(Q, Kk, self.qk_scale)
            scale = 1.0
        else:
            scale = None
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=Q.dtype)
        out = F.scaled_dot_product_attention(
            Q, Kk, V, attn_mask=attn_mask,
            dropout_p=self.drop if self.training else 0.0, scale=scale,
        )
        out = self.Wo(out.transpose(1, 2).reshape(B, K, D))
        return x + self.res_scale * out


class OuterGatherCycle(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.d = d_model // n_heads; self.drop = dropout
        self.use_qk_norm = use_qk_norm
        self.norm_q = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        self.res_attn = nn.Parameter(torch.ones(1))
        self.res_ffn = nn.Parameter(torch.ones(1))
        if use_qk_norm:
            self.qk_scale = make_qk_scale(n_heads, self.d)

    def forward(self, inner, K_outer, V_outer, attn_mask=None):
        B, Kl, D = inner.shape
        q = self.norm_q(inner)
        Q = self.Wq(q).view(B, Kl, self.h, self.d).transpose(1, 2)
        if self.use_qk_norm:
            Q, K_outer = apply_qk_norm(Q, K_outer, self.qk_scale)
            scale = 1.0
        else:
            scale = None
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=Q.dtype)
        out = F.scaled_dot_product_attention(
            Q, K_outer, V_outer, attn_mask=attn_mask,
            dropout_p=self.drop if self.training else 0.0, scale=scale,
        )
        out = self.Wo(out.transpose(1, 2).reshape(B, Kl, D))
        gate = torch.sigmoid(self.gate(q))
        inner = inner + self.res_attn * gate * out
        return inner + self.res_ffn * self.ffn(self.ffn_norm(inner))


class InnerGatherCycle(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.d = d_model // n_heads; self.drop = dropout
        self.use_qk_norm = use_qk_norm
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        self.res_attn = nn.Parameter(torch.ones(1))
        self.res_ffn = nn.Parameter(torch.ones(1))
        if use_qk_norm:
            self.qk_scale = make_qk_scale(n_heads, self.d)

    def project_kv(self, kv_src):
        B = kv_src.shape[0]
        Nkv = kv_src.shape[1]
        kv = self.norm_kv(kv_src)
        K = self.Wk(kv).view(B, Nkv, self.h, self.d).transpose(1, 2)
        V = self.Wv(kv).view(B, Nkv, self.h, self.d).transpose(1, 2)
        return K, V

    def forward(self, q_src, kv_src=None, routing_k=None, kv_cache=None, attn_mask=None):
        B, Nq, D = q_src.shape
        q = self.norm_q(q_src)
        Q = self.Wq(q).view(B, Nq, self.h, self.d).transpose(1, 2)

        if kv_cache is not None:
            K, V = kv_cache
        else:
            K, V = self.project_kv(kv_src)
        Nkv = K.shape[2]

        if self.use_qk_norm:
            Q, K = apply_qk_norm(Q, K, self.qk_scale)
            sdpa_scale = 1.0
            manual_scale = 1.0
        else:
            sdpa_scale = None
            manual_scale = 1.0 / math.sqrt(self.d)

        if routing_k is not None and Nkv > routing_k:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * manual_scale
            if attn_mask is not None:
                scores = scores + attn_mask
            threshold = torch.topk(scores, routing_k, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.training and self.drop > 0.0:
                attn = F.dropout(attn, p=self.drop)
            out = torch.matmul(attn, V)
        else:
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask,
                dropout_p=self.drop if self.training else 0.0, scale=sdpa_scale,
            )
        out = self.Wo(out.transpose(1, 2).reshape(B, Nq, D))
        gate = torch.sigmoid(self.gate(q))
        q_src = q_src + self.res_attn * gate * out
        return q_src + self.res_ffn * self.ffn(self.ffn_norm(q_src))

class OuterDecodeCycle(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_qk_norm=False):
        super().__init__()
        self.h = n_heads; self.d = d_model // n_heads; self.drop = dropout
        self.use_qk_norm = use_qk_norm
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        self.res_attn = nn.Parameter(torch.ones(1))
        self.res_ffn = nn.Parameter(torch.ones(1))
        if use_qk_norm:
            self.qk_scale = make_qk_scale(n_heads, self.d)

    def forward(self, outer_q, inner_kv, decode_mask=None):
        B, T, d = outer_q.shape
        K_len = inner_kv.shape[1]
        q = self.norm_q(outer_q)
        kv = self.norm_kv(inner_kv)
        Q = self.Wq(q).view(B, T, self.h, self.d).transpose(1, 2)
        Kk = self.Wk(kv).view(B, K_len, self.h, self.d).transpose(1, 2)
        V = self.Wv(kv).view(B, K_len, self.h, self.d).transpose(1, 2)
        if self.use_qk_norm:
            Q, Kk = apply_qk_norm(Q, Kk, self.qk_scale)
            scale = 1.0
        else:
            scale = None
        if decode_mask is not None:
            decode_mask = decode_mask.to(dtype=Q.dtype)
        out = F.scaled_dot_product_attention(
            Q, Kk, V, attn_mask=decode_mask,
            dropout_p=self.drop if self.training else 0.0, scale=scale,
        )
        out = self.Wo(out.transpose(1, 2).reshape(B, T, d))
        outer_q = outer_q + self.res_attn * out
        return outer_q + self.res_ffn * self.ffn(self.ffn_norm(outer_q))

class RadialCompactionRing(nn.Module):
    def __init__(self, d_model, d_compact=None, use_rezero=True, rezero_init=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_compact = d_compact or max(8, d_model // 4)
        self.compressor = nn.Linear(d_model, self.d_compact, bias=False)
        self.decompressor = nn.Linear(self.d_compact, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.res_scale = nn.Parameter(torch.ones(1) * (rezero_init if use_rezero else 1.0))

    def forward(self, x):
        x_norm = self.norm(x)
        compressed = self.compressor(x_norm)
        decompressed = self.decompressor(compressed)
        rec_loss = F.mse_loss(decompressed, x_norm)
        if self.training:
            W_comp = self.compressor.weight
            try:
                S = torch.linalg.svdvals(W_comp)
                rank_loss = torch.sum(S)
            except RuntimeError:
                rank_loss = torch.sum(W_comp ** 2)
        else:
            rank_loss = x.new_zeros(())
        compaction_loss = rec_loss + 1e-3 * rank_loss
        x_out = x + self.res_scale * (decompressed - x)
        return x_out, compaction_loss

class LatentMerger(nn.Module):
    def __init__(self, d_model, K, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.K = K; self.d_model = d_model; self.h = n_heads; self.d = d_model // n_heads
        self.queries = nn.Parameter(torch.randn(1, K, d_model) * 0.02)
        self.older_marker = nn.Parameter(torch.zeros(1, 1, d_model))
        self.newer_marker = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff_g = nn.Linear(d_model, d_model * 2, bias=False)
        self.ff_p = nn.Linear(d_model, d_model * 2, bias=False)
        self.ff_out = nn.Linear(d_model * 2, d_model, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.ff_out.weight)

    def forward(self, latents):
        B = latents.shape[0]
        q0 = self.queries.expand(B, -1, -1)
        older = latents[:, :self.K] + self.older_marker
        newer = latents[:, self.K:] + self.newer_marker
        marked = torch.cat([older, newer], dim=1)
        q = self.norm_q(q0)
        kv = self.norm_kv(marked)
        Q = self.Wq(q).view(B, self.K, self.h, self.d).transpose(1, 2)
        K = self.Wk(kv).view(B, 2 * self.K, self.h, self.d).transpose(1, 2)
        V = self.Wv(kv).view(B, 2 * self.K, self.h, self.d).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = self.Wo(out.transpose(1, 2).reshape(B, self.K, -1))
        r = q0 + out
        n = self.norm_ff(r)
        return r + self.ff_out(F.silu(self.ff_g(n)) * self.ff_p(n))


class BinaryTemporalTree:
    def __init__(self, d_model, K, merger, max_levels=40, device='cpu', dtype=torch.float16):
        self.K = K; self.d_model = d_model; self.merger = merger; self.max_L = max_levels
        self.device = device; self.dtype = dtype
        self.n_chunks = 0; self.B = [None] * max_levels; self._push_count = 0

    @staticmethod
    def _chunk_pos_encoding(chunk_id, d_model, device, dtype):
        half = d_model // 2
        i = torch.arange(half, device=device, dtype=torch.float32)
        freq = 1.0 / (10000 ** (i / half))
        enc = torch.zeros(d_model, device=device, dtype=torch.float32)
        enc[:half] = torch.sin(chunk_id * freq)
        enc[half:] = torch.cos(chunk_id * freq)
        return (enc * 0.01).to(dtype).reshape(1, 1, d_model)

    @torch.no_grad()
    def push(self, latents):
        chunk_id = self._push_count
        self._push_count += 1
        d = latents.shape[-1]
        pos_signal = self._chunk_pos_encoding(chunk_id, d, latents.device, latents.dtype)
        marked_latents = latents + pos_signal
        carry = marked_latents.detach().to(self.dtype).cpu()
        for level in range(self.max_L):
            if self.B[level] is None:
                self.B[level] = carry
                self.n_chunks += 1
                return
            older = self.B[level].to(self.device).float()
            newer = carry.to(self.device).float()
            merged = self.merger(torch.cat([older, newer], dim=1))
            carry = merged.detach().to(self.dtype).cpu()
            self.B[level] = None
        raise OverflowError(f"BTT overflow: all {self.max_L} levels occupied.")

    def get_bank(self):
        occupied = sorted([(i, b) for i, b in enumerate(self.B) if b is not None], key=lambda x: -x[0])
        if not occupied:
            return None
        tensors = [b.to(self.device).float() for _, b in occupied]
        return torch.cat(tensors, dim=1)

    @property
    def occupied_levels(self):
        return [i for i, b in enumerate(self.B) if b is not None]

    @property
    def total_latents(self):
        return len(self.occupied_levels) * self.K

    @property
    def coverage_chunks(self):
        return sum(2**i for i in self.occupied_levels)

    def stats(self):
        lvls = self.occupied_levels
        mb = self.total_latents * self.d_model * 2 / 1e6
        cov_t = self.coverage_chunks * 16384
        return (f"BTT │ pushed={self._push_count} "
                f"compressed_chunks={self.n_chunks} "
                f"levels={lvls} latents={self.total_latents} "
                f"≈{mb:.2f}MB covers≈{cov_t/1e6:.1f}M tokens")

    def save(self, path):
        torch.save({
            'B': self.B, 'K': self.K, 'd_model': self.d_model,
            'n_chunks': self.n_chunks, 'max_L': self.max_L,
            'push_count': self._push_count,
        }, path)

    @classmethod
    def load(cls, path, merger, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        obj = cls(state['d_model'], state['K'], merger, max_levels=state.get('max_L', 40), device=device)
        obj.B = state['B']
        obj.n_chunks = state['n_chunks']
        obj._push_count = state.get('push_count', obj.n_chunks)
        return obj

class TurboQuantizer:
    def __init__(self, d_head, n_heads, device='cpu', dtype=torch.float16):
        gen = torch.Generator()
        gen.manual_seed(42)
        H_mat = torch.randn(d_head, d_head, dtype=torch.float32, generator=gen)
        Q, _ = torch.linalg.qr(H_mat)
        self.rotation_matrix = Q.to(device=device, dtype=dtype)
        self.n_heads = n_heads
        self.q_max = 127.0
        # per-channel scale: (1, H, 1, dh)
        self.scale_k = torch.ones(1, n_heads, 1, d_head, device=device, dtype=torch.float32)
        self.scale_v = torch.ones(1, n_heads, 1, d_head, device=device, dtype=torch.float32)
        self.n_recalibrations = 0

    def calibrate(self, k: torch.Tensor, v: torch.Tensor, headroom: float = 1.15):
        k_rot = torch.matmul(k.to(self.rotation_matrix.dtype), self.rotation_matrix)
        v_rot = torch.matmul(v.to(self.rotation_matrix.dtype), self.rotation_matrix)
        self.scale_k = (k_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
                        * headroom / self.q_max).reshape(1, -1, 1, k_rot.shape[-1])
        self.scale_v = (v_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
                        * headroom / self.q_max).reshape(1, -1, 1, v_rot.shape[-1])

    @torch.no_grad()
    def check_and_expand(self, k: torch.Tensor, v: torch.Tensor):
        old_k = self.scale_k.max().item()
        old_v = self.scale_v.max().item()
        k_rot = torch.matmul(k.to(self.rotation_matrix.dtype), self.rotation_matrix)
        v_rot = torch.matmul(v.to(self.rotation_matrix.dtype), self.rotation_matrix)
        k_max = k_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
        v_max = v_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
        self.scale_k = torch.maximum(
            self.scale_k.to(k_max.device),
            (k_max / self.q_max).reshape(1, -1, 1, k_rot.shape[-1]),
        )
        self.scale_v = torch.maximum(
            self.scale_v.to(v_max.device),
            (v_max / self.q_max).reshape(1, -1, 1, v_rot.shape[-1]),
        )
        if self.scale_k.max().item() > old_k or self.scale_v.max().item() > old_v:
            self.n_recalibrations += 1

    def _rot(self, x): return torch.matmul(x.to(self.rotation_matrix.dtype), self.rotation_matrix)
    def _unrot(self, x): return torch.matmul(x, self.rotation_matrix.t())

    def compress_k(self, x):
        s = self.scale_k.to(x.device)
        return torch.clamp(torch.round(self._rot(x) / s), -self.q_max, self.q_max).to(torch.int8)
    def compress_v(self, x):
        s = self.scale_v.to(x.device)
        return torch.clamp(torch.round(self._rot(x) / s), -self.q_max, self.q_max).to(torch.int8)
    def decompress_k(self, q, out_dtype):
        s = self.scale_k.to(q.device)
        return self._unrot(q.to(self.rotation_matrix.dtype) * s).to(out_dtype)
    def decompress_v(self, q, out_dtype):
        s = self.scale_v.to(q.device)
        return self._unrot(q.to(self.rotation_matrix.dtype) * s).to(out_dtype)


class GenerationCache:
    __slots__ = ('T_max', 'H', 'dh', 'D', 'k_minus_1', 'device', 'dtype', 'filled', 'ptr',
                 'K_raw_q', 'V_q', 'emb_win', 'quantizer', '_wrap_warned')
    def __init__(self, T_max, n_heads, d_head, d_model, k_minus_1, batch_size=1, device='cpu', dtype=torch.float16):
        self.T_max = T_max; self.H = n_heads; self.dh = d_head; self.D = d_model; self.k_minus_1 = k_minus_1
        self.device = device; self.dtype = dtype
        self.filled = 0; self.ptr = 0
        self._wrap_warned = False
        self.quantizer = TurboQuantizer(d_head, n_heads, device=device, dtype=dtype)
        B = batch_size
        self.K_raw_q = torch.empty(B, n_heads, T_max, d_head, device=device, dtype=torch.int8)
        self.V_q     = torch.empty(B, n_heads, T_max, d_head, device=device, dtype=torch.int8)
        self.emb_win = torch.zeros(B, k_minus_1, d_model, device=device, dtype=dtype)

    def populate(self, K_raw_full, V_full, emb_last):
        import warnings
        assert K_raw_full.shape[2] == V_full.shape[2], "K and V must be the same length"
        T_src = K_raw_full.shape[2]
        if T_src > self.T_max:
            warnings.warn(
                f"[QCache] Prompt KV length {T_src} exceeds cache capacity "
                f"{self.T_max}; keeping the LAST {self.T_max} positions "
                f"(sliding window). Older context is DROPPED.",
                stacklevel=2,
            )
            K_src = K_raw_full[:, :, -self.T_max:]
            V_src = V_full[:, :, -self.T_max:]
        else:
            K_src = K_raw_full
            V_src = V_full
        T = K_src.shape[2]

        self.quantizer.calibrate(K_src, V_src)
        self.K_raw_q[:, :, :T].copy_(self.quantizer.compress_k(K_src))
        self.V_q[:, :, :T].copy_(self.quantizer.compress_v(V_src))
        self.emb_win.copy_(emb_last.to(self.dtype))
        self.filled = T
        self.ptr = T % self.T_max

    @torch.no_grad()
    def step(self, k_raw_new, v_new, emb_new):
        import warnings
        pos = self.ptr
        if self.filled == self.T_max and not self._wrap_warned:
            warnings.warn(
                f"[QCache] Ring buffer full ({self.T_max}); "
                f"oldest tokens are being evicted (sliding window). "
                f"Consider wiring eviction into BTT for long-context retention.",
                stacklevel=2,
            )
            self._wrap_warned = True
        self.quantizer.check_and_expand(k_raw_new, v_new)
        k_q = self.quantizer.compress_k(k_raw_new)
        v_q = self.quantizer.compress_v(v_new)
        self.K_raw_q[:, :, pos:pos+1].copy_(k_q)
        self.V_q[:, :, pos:pos+1].copy_(v_q)
        self.ptr = (self.ptr + 1) % self.T_max
        if self.filled < self.T_max:
            self.filled += 1
        self.emb_win = torch.cat([self.emb_win[:, 1:], emb_new.to(self.dtype)], dim=1)

    def get_kv(self, model_dtype):
        T = self.filled
        if T == self.T_max:
            idx = (torch.arange(T, device=self.device) + self.ptr) % self.T_max
            K_raw = self.quantizer.decompress_k(self.K_raw_q[:, :, idx], model_dtype)
            V = self.quantizer.decompress_v(self.V_q[:, :, idx], model_dtype)
        else:
            K_raw = self.quantizer.decompress_k(self.K_raw_q[:, :, :T], model_dtype)
            V = self.quantizer.decompress_v(self.V_q[:, :, :T], model_dtype)
        return K_raw, V

class HierarchicalRadialTransformerV7(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model; H = cfg.n_outer_heads; K = cfg.n_outer_latents
        T = cfg.max_seq_len; nc = cfg.n_center_latents; qkn = cfg.use_qk_norm

        self.active_latents = K
        self._mask_cache = OrderedDict()
        self._mask_cache_bytes = int(self.cfg.mask_cache_mb * 1e6)
        self._mask_cache_used_bytes = 0

        self.register_buffer("last_internalization_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_compaction_loss", torch.tensor(0.0), persistent=False)
        self.last_inference_steps = 0

        if cfg.vocab_size > 0:
            self.tok_emb = nn.Embedding(cfg.vocab_size, d)
            self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        else:
            self.tok_emb = None
            self.lm_head = None

        self.emb_drop = nn.Dropout(cfg.dropout)
        self.local_conv = LocalConv(d, cfg.local_conv_kernel, cfg.dropout, causal=cfg.causal)

        H_max = max(cfg.harp_max_heads, H, cfg.n_inner_heads, cfg.n_latent_heads)
        self.harp = HARP(n_heads_max=H_max, K=K, max_seq_len=cfg.max_seq_len * 4, tl_scale=cfg.harp_tl_scale)

        buds = torch.tensor([((i + 1) * T) // K for i in range(K)], dtype=torch.long)
        self.register_buffer("buds", buds)
        l_mask = torch.triu(torch.full((K, K), float("-inf")), diagonal=1)
        self.register_buffer("latent_causal_mask", l_mask[None, None, :, :])

        self.outer_kv_norm = nn.LayerNorm(d)
        self.outer_Wk = nn.Linear(d, d, bias=False)
        self.outer_Wv = nn.Linear(d, d, bias=False)
        self.outer_null_kv = nn.Parameter(torch.zeros(1, 1, d))
        self.inner_latents_base = nn.Parameter(torch.zeros(1, K, d))
        self.relative_time_bias = nn.Parameter(torch.zeros(1, K, d))

        self.outer_cycles = nn.ModuleList([OuterGatherCycle(d, H, cfg.d_ff, cfg.dropout, qkn) for _ in range(cfg.n_outer_cycles)])
        self.latent_self_attn = LatentSelfAttention(d, cfg.n_latent_heads, cfg.dropout, qkn)

        if cfg.use_compaction:
            self.compaction_ring = RadialCompactionRing(d, d_compact=cfg.d_compact, use_rezero=cfg.use_rezero, rezero_init=cfg.rezero_init)

        self.center_base = nn.Parameter(torch.zeros(1, nc, d))
        self.reasoning_core = InnerGatherCycle(d, cfg.n_inner_heads, cfg.d_ff, cfg.dropout, qkn)
        self.broadcast_cycles = nn.ModuleList([InnerGatherCycle(d, cfg.n_inner_heads, cfg.d_ff, cfg.dropout, qkn) for _ in range(cfg.n_broadcast_cycles)])
        self.final_inner_cycle = InnerGatherCycle(d, cfg.n_inner_heads, cfg.d_ff, cfg.dropout, qkn)
        self.final_broadcast = InnerGatherCycle(d, cfg.n_inner_heads, cfg.d_ff, cfg.dropout, qkn)

        self.decode_cycles = nn.ModuleList([OuterDecodeCycle(d, H, cfg.d_ff, cfg.dropout, qkn) for _ in range(cfg.n_outer_cycles)])
        self.decode_norm = nn.LayerNorm(d)
        self.out_norm = nn.LayerNorm(d)
        self.null_latent = nn.Parameter(torch.zeros(1, 1, d))

        if cfg.use_ib:
            self.bridge_head_l0 = nn.Linear(d, cfg.vocab_size, bias=False)
            self.entropy_scale = nn.Parameter(torch.tensor(2.0))
            self.entropy_threshold = nn.Parameter(torch.tensor(2.0))
        else:
            self.bridge_head_l0 = None
            self.entropy_scale = None
            self.entropy_threshold = None

        self.last_fast_logits = None
        self.last_gate = None

        self._init_weights()
        self._apply_rezero()
        self._maybe_compile()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
        nn.init.normal_(self.relative_time_bias, std=0.02)
        nn.init.normal_(self.inner_latents_base, std=0.02)
        nn.init.normal_(self.center_base, std=0.02)
        nn.init.zeros_(self.null_latent)
        nn.init.zeros_(self.outer_null_kv)

    def _apply_rezero(self):
        if not self.cfg.use_rezero: return
        v = self.cfg.rezero_init
        for name, p in self.named_parameters():
            if name.split(".")[-1] in ("res_attn", "res_ffn", "res_scale"):
                with torch.no_grad(): p.fill_(v)

    def _maybe_compile(self):
        if not self.cfg.use_compile: return
        mode = self.cfg.compile_mode
        try:
            self.local_conv = torch.compile(self.local_conv, mode=mode)
            self.latent_self_attn = torch.compile(self.latent_self_attn, mode=mode)
            self.outer_cycles = nn.ModuleList([torch.compile(c, mode=mode) for c in self.outer_cycles])
            self.decode_cycles = nn.ModuleList([torch.compile(c, mode=mode) for c in self.decode_cycles])
        except Exception as e:
            print(f"[HRT] torch.compile failed: {e}")

    def set_active_latents(self, k):
        self.active_latents = max(1, min(int(k), self.cfg.n_outer_latents))

    def _build_outer_hard(self, T, Kc, device):
        if not self.cfg.causal:
            return torch.zeros(1, 1, Kc, T, device=device)
        base_T = T
        i = torch.arange(Kc, device=device)
        buds = (((i + 1).float() * base_T) / Kc).ceil().long().clamp(1, base_T)
        j = torch.arange(base_T, device=device)
        mask = torch.where(j[None] < buds[:, None], torch.zeros(Kc, base_T, device=device), torch.full((Kc, base_T), float("-inf"), device=device))
        return mask[None, None, :, :]

    def _build_decode_hard(self, T, Kc, device):
        if not self.cfg.causal:
            return torch.zeros(1, 1, T, Kc, device=device)
        base_T = T
        i = torch.arange(Kc, device=device)
        buds = (((i + 1).float() * base_T) / Kc).ceil().long().clamp(1, base_T)
        t = torch.arange(base_T, device=device)
        mask = torch.where(buds[None, :] - 1 <= t[:, None], torch.zeros(base_T, Kc, device=device), torch.full((base_T, Kc), float("-inf"), device=device))
        return mask[None, None, :, :]

    def _mask_nbytes(self, kind, T, Kc):
        return T * Kc * 4

    def _cached_mask(self, kind, T, device):
        Kc = self.active_latents
        key = (kind, T, Kc, self.cfg.causal, device.type, getattr(device, "index", -1))
        m = self._mask_cache.get(key)
        if m is not None:
            self._mask_cache.move_to_end(key)  
            return m
        m = (self._build_outer_hard(T, Kc, device) if kind == "outer"
             else self._build_decode_hard(T, Kc, device))
        if self._mask_cache_bytes <= 0:
            return m
        nbytes = self._mask_nbytes(kind, T, Kc)
        if nbytes > self._mask_cache_bytes:
            return m
        self._mask_cache[key] = m
        self._mask_cache_used_bytes += nbytes
        while self._mask_cache_used_bytes > self._mask_cache_bytes:
            _, evicted = self._mask_cache.popitem(last=False)
            self._mask_cache_used_bytes -= evicted.numel() * evicted.element_size()
        return m

    def _get_outer_mask(self, T, device):
        return self._cached_mask("outer", T, device)

    def _get_decode_mask(self, T, device):
        return self._cached_mask("decode", T, device)

    def _get_outer_attn(self, T, device, dtype, pad_bias=None):
        Kc = self.active_latents; H = self.cfg.n_outer_heads
        hard = self._get_outer_mask(T, device).to(dtype=dtype)
        base_T = T
        harp = self.harp.latent_to_token(T, H, device, dtype, K=Kc, base_T=base_T)
        attn = hard + harp
        if pad_bias is not None:
            attn = attn + pad_bias.to(dtype=dtype)
        return attn

    def _get_latent_attn(self, device, dtype):
        Kc = self.active_latents; H = self.cfg.n_latent_heads
        harp = self.harp.latent_to_latent(H, device, dtype, K=Kc)
        if not self.cfg.causal: return harp
        causal = self.latent_causal_mask[:, :, :Kc, :Kc].to(device=device, dtype=dtype)
        return causal + harp

    def _add_outer_sink(self, K_h, V_h, attn, B, H, device, dtype):
        d_h = self.cfg.d_model // H
        null = self.outer_null_kv.expand(B, -1, -1)
        nk = self.outer_Wk(null).view(B, 1, H, d_h).transpose(1, 2)
        nv = self.outer_Wv(null).view(B, 1, H, d_h).transpose(1, 2)
        K_h = torch.cat([nk, K_h], dim=2)
        V_h = torch.cat([nv, V_h], dim=2)
        if attn is not None:
            sink_col = torch.zeros(*attn.shape[:-1], 1, device=device, dtype=dtype)
            attn = torch.cat([sink_col, attn], dim=-1)
        return K_h, V_h, attn

    def _make_decode_masks(self, T, device, ref_dtype, latent_bank=None, B=1, single_step=False):
        Kc = self.active_latents; H = self.cfg.n_outer_heads
        decode_hard = self._get_decode_mask(T, device).to(dtype=ref_dtype)
        base_T = T
        harp_t2l = self.harp.token_to_latent(T, H, device, ref_dtype, K=Kc, base_T=base_T)
        decode_mask = decode_hard + harp_t2l
        if single_step: decode_mask = decode_mask[:, :, -1:, :]
        T_q = 1 if single_step else T
        null_col = torch.zeros(1, H, T_q, 1, device=device, dtype=ref_dtype)
        if latent_bank is not None:
            bank_len = latent_bank.shape[1]
            bank_mask = torch.zeros(1, H, T_q, bank_len, device=device, dtype=ref_dtype)
            full_mask = torch.cat([null_col, bank_mask, decode_mask], dim=-1)
        else:
            full_mask = torch.cat([null_col, decode_mask], dim=-1)
        return full_mask

    def _build_outer_kv(self, outer_n, B, T, H, d_h):
        chunk = self.cfg.proj_chunk
        if chunk and T > chunk:
            Ks, Vs = [], []
            for i in range(0, T, chunk):
                c = outer_n[:, i: i + chunk]
                Ks.append(self.outer_Wk(c)); Vs.append(self.outer_Wv(c))
            K_f = torch.cat(Ks, 1); V_f = torch.cat(Vs, 1)
        else:
            K_f = self.outer_Wk(outer_n); V_f = self.outer_Wv(outer_n)
        K_h = K_f.view(B, T, H, d_h).transpose(1, 2)
        V_h = V_f.view(B, T, H, d_h).transpose(1, 2)
        return K_h, V_h

    def _run_inner(self, outer, T, device, latent_bank=None, padding_mask=None):
        cfg = self.cfg; B = outer.shape[0]; H = cfg.n_outer_heads; d_h = cfg.d_model // H; rk = cfg.routing_k
        ref_dtype = outer.dtype; Kc = self.active_latents

        outer_n = self.outer_kv_norm(outer)
        K_h, V_h = self._build_outer_kv(outer_n, B, T, H, d_h)

        pb = None
        if padding_mask is not None:
            pb = torch.zeros(B, 1, 1, T, device=device, dtype=ref_dtype)
            pb.masked_fill_(~padding_mask[:, None, None, :], float("-inf"))

        attn = self._get_outer_attn(T, device, ref_dtype, pad_bias=pb)
        K_h_raw, V_h_raw = K_h, V_h
        K_h, V_h, attn = self._add_outer_sink(K_h, V_h, attn, B, H, device, ref_dtype)
        lat_attn = self._get_latent_attn(device, ref_dtype)

        inner = self.inner_latents_base[:, :Kc].expand(B, -1, -1).clone() + self.relative_time_bias[:, :Kc]
        center = self.center_base.expand(B, -1, -1).clone()

        ckpt = cfg.use_grad_ckpt and self.training
        for cyc in self.outer_cycles:
            if ckpt: inner = checkpoint.checkpoint(cyc, inner, K_h, V_h, attn, use_reentrant=False)
            else: inner = cyc(inner, K_h, V_h, attn)
            inner = self.latent_self_attn(inner, attn_mask=lat_attn)

        if cfg.use_compaction:
            inner, comp_loss = self.compaction_ring(inner)
            self.last_compaction_loss = comp_loss
        else:
            self.last_compaction_loss = torch.zeros((), device=device, dtype=ref_dtype)

        full_ctx = torch.cat([latent_bank, inner], dim=1) if latent_bank is not None else inner
        kv_cache = self.reasoning_core.project_kv(full_ctx)
        center = self._run_reasoning_core(center, kv_cache, rk)

        inner_upd = inner
        for b_cyc in self.broadcast_cycles:
            inner_upd = b_cyc(inner_upd, center, routing_k=rk)

        full_ctx_f = torch.cat([latent_bank, inner_upd], dim=1) if latent_bank is not None else inner_upd
        center = self.final_inner_cycle(center, full_ctx_f, routing_k=rk)
        inner_upd = self.final_broadcast(inner_upd, center, routing_k=rk)

        return inner, inner_upd, center, K_h_raw, V_h_raw
        
    def _run_reasoning_core(self, center, kv_cache, rk):
        cfg = self.cfg
        ref_dtype = center.dtype
        device = center.device
        self.last_internalization_loss = torch.zeros((), device=device, dtype=ref_dtype)

        if not self.training and cfg.use_adaptive_halting:
            step, max_steps, eps = 0, cfg.max_adaptive_steps, cfg.adaptive_epsilon
            while step < max_steps:
                prev = center
                center = self.reasoning_core(center, kv_cache=kv_cache, routing_k=rk)
                step += 1
                if torch.mean((center - prev) ** 2).item() < eps:
                    break
            self.last_inference_steps = step
            return center

        if self.training and cfg.use_internalization:
            n_steps = cfg.internalization_steps
            if cfg.cycle_drop:
                n_steps = random.randint(2, n_steps)
            if cfg.use_jfb:
                with torch.no_grad():
                    z = center
                    for _ in range(n_steps - 1):
                        z = self.reasoning_core(z, kv_cache=kv_cache, routing_k=rk)
                z = z.detach()
                z_star = self.reasoning_core(z, kv_cache=kv_cache, routing_k=rk)
                self.last_internalization_loss = torch.mean((z_star - z) ** 2)
                return z_star
            else:
                center_1 = self.reasoning_core(center, kv_cache=kv_cache, routing_k=rk)
                current = center_1
                loss = 0.0
                for _ in range(n_steps - 2):
                    nxt = self.reasoning_core(current, kv_cache=kv_cache, routing_k=rk)
                    loss = loss + torch.mean((nxt - current) ** 2)
                    current = nxt
                final = self.reasoning_core(current, kv_cache=kv_cache, routing_k=rk)
                loss = loss + torch.mean((final - current) ** 2)
                self.last_internalization_loss = loss / (n_steps - 1)
                return final

        n_cycles = cfg.n_inner_cycles
        if self.training and cfg.cycle_drop:
            n_cycles = random.randint(cfg.min_inner_cycles, cfg.n_inner_cycles)
        self.last_inference_steps = n_cycles
        for _ in range(n_cycles):
            center = self.reasoning_core(center, kv_cache=kv_cache, routing_k=rk)
        return center

    def forward(self, x, padding_mask=None, latent_bank=None, segment_ids=None):
        B, T, _ = x.shape
        device = x.device
        cfg = self.cfg
        ref_dtype = x.dtype

        outer = self.emb_drop(x)
        outer = self.local_conv(outer, segment_ids=segment_ids)

        inner, inner_upd, center, K_h, V_h = self._run_inner(outer, T, device, latent_bank, padding_mask=padding_mask)

        if self.bridge_head_l0 is not None:
            self.last_fast_logits = self.bridge_head_l0(outer)

        full_mask_nd = self._make_decode_masks(T, device, ref_dtype, latent_bank=latent_bank, B=B)
        null_exp = self.null_latent.expand(B, -1, -1)
        if latent_bank is not None:
            full_latents_nd = torch.cat([null_exp, latent_bank, inner_upd], dim=1)
        else:
            full_latents_nd = torch.cat([null_exp, inner_upd], dim=1)

        out_f = outer

        gate = None
        if self.bridge_head_l0 is not None:
            probs = F.softmax(self.last_fast_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(self.entropy_scale * (entropy - self.entropy_threshold))
            self.last_gate = gate

        for cyc in self.decode_cycles:
            out_f = cyc(out_f, full_latents_nd, full_mask_nd)

        if gate is not None:
            out_f = outer + gate * (out_f - outer)

        out_f_n = self.decode_norm(out_f)

        if cfg.return_internalization_loss:
            total_aux_loss = self.last_internalization_loss + self.last_compaction_loss * cfg.compaction_loss_weight
            return out_f_n, total_aux_loss
        return out_f_n

    @torch.no_grad()
    def extract_latents(self, x, padding_mask=None, segment_ids=None):
        B, T, _ = x.shape
        device = x.device

        outer = self.local_conv(self.emb_drop(x), segment_ids=segment_ids)
        inner, _, _, _, _ = self._run_inner(outer, T, device, None, padding_mask=padding_mask)
        return inner

    @torch.no_grad()
    def _init_generation_cache(self, x, latent_bank=None):
        B, T, _ = x.shape
        device = x.device
        cfg = self.cfg
        mdtype = x.dtype
        H, d_h = cfg.n_outer_heads, cfg.d_model // cfg.n_outer_heads
        Kc = self.active_latents

        outer = self.local_conv(self.emb_drop(x))
        _, inner_upd, center, K_h, V_h = self._run_inner(outer, T, device, latent_bank)

        assert K_h.shape[2] == T, (
            f"QCache invariant broken: expected K of length {T}, got {K_h.shape[2]} "
            f"(sink leaked into cache?)"
        )

        k_minus_1 = self.local_conv.k - 1
        if T >= k_minus_1:
            emb_last = x[:, -k_minus_1:]
        else:
            pad = torch.zeros(B, k_minus_1 - T, cfg.d_model, device=device)
            emb_last = torch.cat([pad, x], dim=1)

        cache = GenerationCache(T_max=cfg.max_seq_len, n_heads=H, d_head=d_h,
                                d_model=cfg.d_model, k_minus_1=k_minus_1,
                                batch_size=B, device=device, dtype=mdtype)
        cache.populate(K_h.detach(), V_h.detach(), emb_last.detach())

        ref_dtype = outer.dtype
        full_mask_nd = self._make_decode_masks(T, device, ref_dtype, latent_bank=latent_bank, B=B, single_step=True)
        null_exp = self.null_latent.expand(B, -1, -1)
        if latent_bank is not None:
            full_latents_nd = torch.cat([null_exp, latent_bank, inner_upd], dim=1)
        else:
            full_latents_nd = torch.cat([null_exp, inner_upd], dim=1)

        out_f = outer[:, -1:]
        for cyc in self.decode_cycles:
            out_f = cyc(out_f, full_latents_nd, full_mask_nd)

        if self.bridge_head_l0 is not None:
            fast_logits = self.bridge_head_l0(outer[:, -1:])
            probs = F.softmax(fast_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(self.entropy_scale * (entropy - self.entropy_threshold))
            out_f = outer[:, -1:] + gate * (out_f - outer[:, -1:])

        h_last = self.decode_norm(out_f.squeeze(1)).detach()   # (B, D)
        if self.lm_head is not None:
            return self.lm_head(h_last), cache
        return h_last, cache   # vocab=0

    @torch.no_grad()
    def _forward_incremental(self, nxt_emb, cache, latent_bank=None):
        B = nxt_emb.shape[0]
        device = nxt_emb.device
        cfg = self.cfg
        H, d_h = cfg.n_outer_heads, cfg.d_model // cfg.n_outer_heads
        mdtype = nxt_emb.dtype
        Kc = self.active_latents

        outer_new = self.local_conv.forward_last(nxt_emb, cache.emb_win.float())
        outer_n_new = self.outer_kv_norm(outer_new)
        k_new = self.outer_Wk(outer_n_new).view(B, 1, H, d_h).transpose(1, 2)
        v_new = self.outer_Wv(outer_n_new).view(B, 1, H, d_h).transpose(1, 2)

        K_h_old, V_h_old = cache.get_kv(mdtype)

        K_h_full = torch.cat([K_h_old, k_new], dim=2)
        V_h_full = torch.cat([V_h_old, v_new], dim=2)

        T_cur = cache.filled + 1

        harp_b = self.harp.latent_to_token(T_cur, H, device, mdtype, K=Kc, base_T=T_cur)

        hard = self._build_outer_hard(T_cur, Kc, device).to(dtype=mdtype)
        attn = hard + harp_b

        K_h_full, V_h_full, attn = self._add_outer_sink(K_h_full, V_h_full, attn, B, H, device, mdtype)
        lat_attn = self._get_latent_attn(device, mdtype)

        inner = self.inner_latents_base[:, :Kc].expand(B, -1, -1).clone() + self.relative_time_bias[:, :Kc]
        center = self.center_base.expand(B, -1, -1).clone()

        for cyc in self.outer_cycles:
            inner = cyc(inner, K_h_full, V_h_full, attn)
            inner = self.latent_self_attn(inner, attn_mask=lat_attn)

        if cfg.use_compaction:
            inner, _ = self.compaction_ring(inner)

        full_ctx = torch.cat([latent_bank, inner], dim=1) if latent_bank is not None else inner
        kv_cache = self.reasoning_core.project_kv(full_ctx)
        center = self._run_reasoning_core(center, kv_cache, cfg.routing_k)

        inner_upd = inner
        for b_cyc in self.broadcast_cycles:
            inner_upd = b_cyc(inner_upd, center, routing_k=cfg.routing_k)

        full_ctx_f = torch.cat([latent_bank, inner_upd], dim=1) if latent_bank is not None else inner_upd
        center = self.final_inner_cycle(center, full_ctx_f, routing_k=cfg.routing_k)
        inner_upd = self.final_broadcast(inner_upd, center, routing_k=cfg.routing_k)

        ref_dtype = mdtype
        full_mask_nd = self._make_decode_masks(T_cur, device, ref_dtype, latent_bank=latent_bank, B=B, single_step=True)
        null_exp = self.null_latent.expand(B, -1, -1)
        if latent_bank is not None:
            full_latents_nd = torch.cat([null_exp, latent_bank, inner_upd], dim=1)
        else:
            full_latents_nd = torch.cat([null_exp, inner_upd], dim=1)

        out_f = outer_new
        for cyc in self.decode_cycles:
            out_f = cyc(out_f, full_latents_nd, full_mask_nd)

        if self.bridge_head_l0 is not None:
            fast_logits = self.bridge_head_l0(outer_new)
            probs = F.softmax(fast_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(self.entropy_scale * (entropy - self.entropy_threshold))
            out_f = outer_new + gate * (out_f - outer_new)

        h = self.decode_norm(out_f.squeeze(1))   # (B, D)

        cache.step(k_new, v_new, nxt_emb)
        if self.lm_head is not None:
            return self.lm_head(h)
        return h   # vocab=0

    @torch.no_grad()
    def generate(self, prompt_emb, max_new, temperature=1.0, top_k=None, latent_bank=None,
                 eos_id=None, use_cache=None, lm_head=None):
        head = self.lm_head if self.lm_head is not None else lm_head
        if head is None:
            raise ValueError(
                "generate() requires a language-model head: either build the model "
                "with vocab_size > 0 or pass an external `lm_head=...` callable. "
                "(HRT v7 primary mode is feature extraction with vocab_size=0, "
                "in which case use step_generation() for cached hidden states.)"
            )
        if use_cache is None:
            use_cache = self.cfg.use_q_cache
        if not use_cache:
            print("[WARN] QCache disabled – full forward each step (slow, memory heavy).")
            gen_emb = prompt_emb
            for step in range(max_new):
                out = self.forward(gen_emb, latent_bank=latent_bank)
                if isinstance(out, tuple):   # return_internalization_loss=True
                    out = out[0]
                logits = head(out[:, -1])
                l = logits / max(temperature, 1e-6)
                if top_k:
                    v, _ = torch.topk(l, min(top_k, l.size(-1)))
                    l[l < v[:, [-1]]] = float('-inf')
                probs = F.softmax(l, dim=-1)
                nxt = torch.multinomial(probs, 1)  # (1,1) – token id
                nxt_emb = self.tok_emb(nxt)        # (1,1,D)
                gen_emb = torch.cat([gen_emb, nxt_emb], dim=1)
                if eos_id is not None and nxt.item() == eos_id:
                    break
            return gen_emb

        out_or_logits, cache = self._init_generation_cache(prompt_emb, latent_bank=latent_bank)
        logits = head(out_or_logits) if self.lm_head is None else out_or_logits
        generated_embs = []
        for _ in range(max_new):
            l = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(l, min(top_k, l.size(-1)))
                l[l < v[:, [-1]]] = float('-inf')
            probs = F.softmax(l, dim=-1)
            nxt = torch.multinomial(probs, 1)       # (1,1)
            nxt_emb = self.tok_emb(nxt)             # (1,1,D)
            generated_embs.append(nxt_emb)          # append eos
            if eos_id is not None and nxt.item() == eos_id:
                break
            inner = self._forward_incremental(nxt_emb, cache, latent_bank=latent_bank)
            # inner — logits, hidden (B, D), vocab=0
            logits = head(inner) if self.lm_head is None else inner

        out_embs = torch.cat([prompt_emb] + generated_embs, dim=1)
        return out_embs
    
    @torch.no_grad()
    def step_generation(self, nxt_emb, cache, latent_bank=None):
        return self._forward_incremental(nxt_emb, cache, latent_bank=latent_bank)


if __name__ == "__main__":
    print("✔")