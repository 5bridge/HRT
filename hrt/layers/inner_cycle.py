import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.qk_norm import apply_qk_norm, make_qk_scale
from ..ops.ffn import SwiGLUFFN


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