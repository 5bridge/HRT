import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.0, causal=True):
        super().__init__()
        self.k = kernel_size
        self.dilation = 3
        self.causal = causal
        self.norm = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=d_model, bias=False)
        self.dw_conv_dilated = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                                         dilation=self.dilation, groups=d_model, bias=False)
        self.pw_conv = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.blend = nn.Parameter(torch.tensor(0.5))
        self.drop = nn.Dropout(dropout)
        self._dilated_pad = (3 - 1) * self.dilation

    def _pad(self, x, total):
        if self.causal:
            return F.pad(x, (total, 0))
        left = total // 2
        right = total - left
        return F.pad(x, (left, right))

    def _causal_dwconv_seg(self, x, weight, dilation, segment_ids):
        B, D, T = x.shape
        Kk = weight.shape[-1]
        seg = segment_ids
        y = x.new_zeros(B, D, T)
        for j in range(Kk):
            shift = (Kk - 1 - j) * dilation
            if shift == 0:
                xj = x
                valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
            else:
                xj = F.pad(x, (shift, 0))[..., :T]
                seg_shift = F.pad(seg, (shift, 0), value=-1)[..., :T]
                valid = seg_shift == seg
            wj = weight[:, 0, j].view(1, D, 1)
            y = y + wj * xj * valid.unsqueeze(1).to(x.dtype)
        return y

    def forward(self, x, segment_ids=None):
        r = x
        x = self.norm(x).transpose(1, 2)
        if segment_ids is None:
            xp = self._pad(x, self.k - 1)
            y1 = F.gelu(self.dw_conv(xp))
            xd = self._pad(x, self._dilated_pad)
            y2 = F.gelu(self.dw_conv_dilated(xd))
        else:
            assert self.causal, "segment-aware conv requires causal=True"
            y1 = F.gelu(self._causal_dwconv_seg(x, self.dw_conv.weight, 1, segment_ids))
            y2 = F.gelu(self._causal_dwconv_seg(x, self.dw_conv_dilated.weight, self.dilation, segment_ids))
        b = torch.sigmoid(self.blend)
        y = self.pw_conv(b * y1 + (1 - b) * y2).transpose(1, 2)
        return r + self.drop(y)

    def forward_last(self, new_emb, emb_win, segment_id=None, emb_win_segment_ids=None):
        if segment_id is not None and emb_win_segment_ids is not None:
            if (emb_win_segment_ids != segment_id).any():
                raise ValueError(
                    "forward_last: segment boundary within the window — "
                    "An incremental path does not support segment intersections. "
                    "At the segment boundary, recreate the cache using _init_generation_cache."
                )
        window = torch.cat([emb_win, new_emb], dim=1)
        r = new_emb
        x = self.norm(window).transpose(1, 2)

        xp = F.pad(x, (self.k - 1, 0))
        y1 = F.gelu(self.dw_conv(xp))[:, :, -1:]

        xd = F.pad(x, (self._dilated_pad, 0))
        y2 = F.gelu(self.dw_conv_dilated(xd))[:, :, -1:]

        b = torch.sigmoid(self.blend)
        y = self.pw_conv(b * y1 + (1 - b) * y2).transpose(1, 2)
        return r + self.drop(y)