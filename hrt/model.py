from __future__ import annotations

import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict

from .config import ModelConfig
from .layers import (
    LocalConv, HARP, LatentSelfAttention,
    OuterGatherCycle, InnerGatherCycle, OuterDecodeCycle,
    RadialCompactionRing,
)
from .cache import GenerationCache


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
                if isinstance(out, tuple):
                    out = out[0]
                logits = head(out[:, -1])
                l = logits / max(temperature, 1e-6)
                if top_k:
                    v, _ = torch.topk(l, min(top_k, l.size(-1)))
                    l[l < v[:, [-1]]] = float('-inf')
                probs = F.softmax(l, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_emb = self.tok_emb(nxt)
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
            nxt = torch.multinomial(probs, 1)
            nxt_emb = self.tok_emb(nxt)
            generated_embs.append(nxt_emb)
            if eos_id is not None and nxt.item() == eos_id:
                break
            inner = self._forward_incremental(nxt_emb, cache, latent_bank=latent_bank)
            logits = head(inner) if self.lm_head is None else inner

        out_embs = torch.cat([prompt_emb] + generated_embs, dim=1)
        return out_embs

    @torch.no_grad()
    def step_generation(self, nxt_emb, cache, latent_bank=None):
        return self._forward_incremental(nxt_emb, cache, latent_bank=latent_bank)