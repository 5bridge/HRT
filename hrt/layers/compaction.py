import torch
import torch.nn as nn
import torch.nn.functional as F


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