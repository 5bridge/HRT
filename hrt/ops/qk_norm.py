import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_qk_norm(Q, K, qk_scale):
    Q = F.normalize(Q, dim=-1)
    K = F.normalize(K, dim=-1)
    g = qk_scale.exp().view(1, -1, 1, 1)
    return Q * g, K


def make_qk_scale(n_heads, head_dim):
    init = math.log(math.sqrt(head_dim))
    return nn.Parameter(torch.full((n_heads,), init, dtype=torch.float32))