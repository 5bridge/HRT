import torch


class BinaryTemporalTree:
    def __init__(self, d_model, K, merger, max_levels=40, device='cpu', dtype=torch.float16):
        self.K = K; self.d_model = d_model; self.merger = merger; self.max_L = max_levels
        self.device = device; self.dtype = dtype
        self.n_chunks = 0; self.B = [None] * max_levels; self._push_count = 0

    @staticmethod
    def _chunk_pos_encoding(chunk_id, d_model, device, dtype):
        half = d_model // 2
        i = torch.arange(half, device=device, dtype=torch.float32)
        freq = 1.0 / (10000 ** (i / half))
        enc = torch.zeros(d_model, device=device, dtype=torch.float32)
        enc[:half] = torch.sin(chunk_id * freq)
        enc[half:] = torch.cos(chunk_id * freq)
        return (enc * 0.01).to(dtype).reshape(1, 1, d_model)

    @torch.no_grad()
    def push(self, latents):
        chunk_id = self._push_count
        self._push_count += 1
        d = latents.shape[-1]
        pos_signal = self._chunk_pos_encoding(chunk_id, d, latents.device, latents.dtype)
        marked_latents = latents + pos_signal
        carry = marked_latents.detach().to(self.dtype).cpu()
        for level in range(self.max_L):
            if self.B[level] is None:
                self.B[level] = carry
                self.n_chunks += 1
                return
            older = self.B[level].to(self.device).float()
            newer = carry.to(self.device).float()
            merged = self.merger(torch.cat([older, newer], dim=1))
            carry = merged.detach().to(self.dtype).cpu()
            self.B[level] = None
        raise OverflowError(f"BTT overflow: all {self.max_L} levels occupied.")

    def get_bank(self):
        occupied = sorted([(i, b) for i, b in enumerate(self.B) if b is not None], key=lambda x: -x[0])
        if not occupied:
            return None
        tensors = [b.to(self.device).float() for _, b in occupied]
        return torch.cat(tensors, dim=1)

    @property
    def occupied_levels(self):
        return [i for i, b in enumerate(self.B) if b is not None]

    @property
    def total_latents(self):
        return len(self.occupied_levels) * self.K

    @property
    def coverage_chunks(self):
        return sum(2**i for i in self.occupied_levels)

    def stats(self):
        lvls = self.occupied_levels
        mb = self.total_latents * self.d_model * 2 / 1e6
        cov_t = self.coverage_chunks * 16384
        return (f"BTT │ pushed={self._push_count} "
                f"compressed_chunks={self.n_chunks} "
                f"levels={lvls} latents={self.total_latents} "
                f"≈{mb:.2f}MB covers≈{cov_t/1e6:.1f}M tokens")

    def save(self, path):
        torch.save({
            'B': self.B, 'K': self.K, 'd_model': self.d_model,
            'n_chunks': self.n_chunks, 'max_L': self.max_L,
            'push_count': self._push_count,
        }, path)

    @classmethod
    def load(cls, path, merger, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        obj = cls(state['d_model'], state['K'], merger, max_levels=state.get('max_L', 40), device=device)
        obj.B = state['B']
        obj.n_chunks = state['n_chunks']
        obj._push_count = state.get('push_count', obj.n_chunks)
        return obj