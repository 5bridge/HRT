import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.qk_norm import apply_qk_norm, make_qk_scale


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