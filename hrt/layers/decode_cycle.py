import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.qk_norm import apply_qk_norm, make_qk_scale
from ..ops.ffn import SwiGLUFFN


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