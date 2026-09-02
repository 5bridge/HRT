import torch
import torch.nn as nn
import torch.nn.functional as F


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