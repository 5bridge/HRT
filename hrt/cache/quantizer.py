import torch


class TurboQuantizer:
    def __init__(self, d_head, n_heads, device='cpu', dtype=torch.float16):
        gen = torch.Generator()
        gen.manual_seed(42)
        H_mat = torch.randn(d_head, d_head, dtype=torch.float32, generator=gen)
        Q, _ = torch.linalg.qr(H_mat)
        self.rotation_matrix = Q.to(device=device, dtype=dtype)
        self.n_heads = n_heads
        self.q_max = 127.0
        # per-channel scale: (1, H, 1, dh)
        self.scale_k = torch.ones(1, n_heads, 1, d_head, device=device, dtype=torch.float32)
        self.scale_v = torch.ones(1, n_heads, 1, d_head, device=device, dtype=torch.float32)
        self.n_recalibrations = 0

    def calibrate(self, k: torch.Tensor, v: torch.Tensor, headroom: float = 1.15):
        k_rot = torch.matmul(k.to(self.rotation_matrix.dtype), self.rotation_matrix)
        v_rot = torch.matmul(v.to(self.rotation_matrix.dtype), self.rotation_matrix)
        self.scale_k = (k_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
                        * headroom / self.q_max).reshape(1, -1, 1, k_rot.shape[-1])
        self.scale_v = (v_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
                        * headroom / self.q_max).reshape(1, -1, 1, v_rot.shape[-1])

    @torch.no_grad()
    def check_and_expand(self, k: torch.Tensor, v: torch.Tensor):
        old_k = self.scale_k.max().item()
        old_v = self.scale_v.max().item()
        k_rot = torch.matmul(k.to(self.rotation_matrix.dtype), self.rotation_matrix)
        v_rot = torch.matmul(v.to(self.rotation_matrix.dtype), self.rotation_matrix)
        k_max = k_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
        v_max = v_rot.abs().amax(dim=(0, 2)).clamp(min=1e-5)
        self.scale_k = torch.maximum(
            self.scale_k.to(k_max.device),
            (k_max / self.q_max).reshape(1, -1, 1, k_rot.shape[-1]),
        )
        self.scale_v = torch.maximum(
            self.scale_v.to(v_max.device),
            (v_max / self.q_max).reshape(1, -1, 1, v_rot.shape[-1]),
        )
        if self.scale_k.max().item() > old_k or self.scale_v.max().item() > old_v:
            self.n_recalibrations += 1

    def _rot(self, x): return torch.matmul(x.to(self.rotation_matrix.dtype), self.rotation_matrix)
    def _unrot(self, x): return torch.matmul(x, self.rotation_matrix.t())

    def compress_k(self, x):
        s = self.scale_k.to(x.device)
        return torch.clamp(torch.round(self._rot(x) / s), -self.q_max, self.q_max).to(torch.int8)

    def compress_v(self, x):
        s = self.scale_v.to(x.device)
        return torch.clamp(torch.round(self._rot(x) / s), -self.q_max, self.q_max).to(torch.int8)

    def decompress_k(self, q, out_dtype):
        s = self.scale_k.to(q.device)
        return self._unrot(q.to(self.rotation_matrix.dtype) * s).to(out_dtype)

    def decompress_v(self, q, out_dtype):
        s = self.scale_v.to(q.device)
        return self._unrot(q.to(self.rotation_matrix.dtype) * s).to(out_dtype)