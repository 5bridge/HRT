import math
import torch
import torch.nn as nn


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