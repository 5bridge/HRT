import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.qk_norm import apply_qk_norm, make_qk_scale
from ..ops.ffn import SwiGLUFFN


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