import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_model, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.out(F.silu(self.gate(x)) * self.proj(x)))