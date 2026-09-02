import warnings
import torch

from .quantizer import TurboQuantizer


class GenerationCache:
    __slots__ = ('T_max', 'H', 'dh', 'D', 'k_minus_1', 'device', 'dtype', 'filled', 'ptr',
                 'K_raw_q', 'V_q', 'emb_win', 'quantizer', '_wrap_warned')

    def __init__(self, T_max, n_heads, d_head, d_model, k_minus_1, batch_size=1, device='cpu', dtype=torch.float16):
        self.T_max = T_max; self.H = n_heads; self.dh = d_head; self.D = d_model; self.k_minus_1 = k_minus_1
        self.device = device; self.dtype = dtype
        self.filled = 0; self.ptr = 0
        self._wrap_warned = False
        self.quantizer = TurboQuantizer(d_head, n_heads, device=device, dtype=dtype)
        B = batch_size
        self.K_raw_q = torch.empty(B, n_heads, T_max, d_head, device=device, dtype=torch.int8)
        self.V_q     = torch.empty(B, n_heads, T_max, d_head, device=device, dtype=torch.int8)
        self.emb_win = torch.zeros(B, k_minus_1, d_model, device=device, dtype=dtype)

    def populate(self, K_raw_full, V_full, emb_last):
        assert K_raw_full.shape[2] == V_full.shape[2], "K and V must be the same length"
        T_src = K_raw_full.shape[2]
        if T_src > self.T_max:
            warnings.warn(
                f"[QCache] Prompt KV length {T_src} exceeds cache capacity "
                f"{self.T_max}; keeping the LAST {self.T_max} positions "
                f"(sliding window). Older context is DROPPED.",
                stacklevel=2,
            )
            K_src = K_raw_full[:, :, -self.T_max:]
            V_src = V_full[:, :, -self.T_max:]
        else:
            K_src = K_raw_full
            V_src = V_full
        T = K_src.shape[2]

        self.quantizer.calibrate(K_src, V_src)
        self.K_raw_q[:, :, :T].copy_(self.quantizer.compress_k(K_src))
        self.V_q[:, :, :T].copy_(self.quantizer.compress_v(V_src))
        self.emb_win.copy_(emb_last.to(self.dtype))
        self.filled = T
        self.ptr = T % self.T_max

    @torch.no_grad()
    def step(self, k_raw_new, v_new, emb_new):
        pos = self.ptr
        if self.filled == self.T_max and not self._wrap_warned:
            warnings.warn(
                f"[QCache] Ring buffer full ({self.T_max}); "
                f"oldest tokens are being evicted (sliding window). "
                f"Consider wiring eviction into BTT for long-context retention.",
                stacklevel=2,
            )
            self._wrap_warned = True
        self.quantizer.check_and_expand(k_raw_new, v_new)
        k_q = self.quantizer.compress_k(k_raw_new)
        v_q = self.quantizer.compress_v(v_new)
        self.K_raw_q[:, :, pos:pos+1].copy_(k_q)
        self.V_q[:, :, pos:pos+1].copy_(v_q)
        self.ptr = (self.ptr + 1) % self.T_max
        if self.filled < self.T_max:
            self.filled += 1
        self.emb_win = torch.cat([self.emb_win[:, 1:], emb_new.to(self.dtype)], dim=1)

    def get_kv(self, model_dtype):
        T = self.filled
        if T == self.T_max:
            idx = (torch.arange(T, device=self.device) + self.ptr) % self.T_max
            K_raw = self.quantizer.decompress_k(self.K_raw_q[:, :, idx], model_dtype)
            V = self.quantizer.decompress_v(self.V_q[:, :, idx], model_dtype)
        else:
            K_raw = self.quantizer.decompress_k(self.K_raw_q[:, :, :T], model_dtype)
            V = self.quantizer.decompress_v(self.V_q[:, :, :T], model_dtype)
        return K_raw, V