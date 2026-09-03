"""
scripts/benchmark_compare_v2.py

More explicit, honest benchmark: HRT vs a vanilla decoder-only Transformer
of matched parameter count, using a realistic config (d_model=512, matching
HRT defaults) and REAL text data (enwik9, byte-level) for the convergence
test instead of synthetic periodic data.

Three benchmarks:
  1. VRAM vs sequence length   (batch=1, fixed d_model) -> tests the
     "64k~6GB / 128k~8GB" claim directly.
  2. VRAM vs d_model            (fixed seq_len) -> tests how memory scales
     with model size, not just context length.
  3. Convergence on real text  (enwik9, byte-level LM) -> loss/throughput
     over a fixed step budget on both models.

Run:
    python scripts/benchmark_compare.py
"""

import gc
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrt import ModelConfig, HierarchicalRadialTransformerV7

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    print("[WARN] No CUDA device found — VRAM numbers will be meaningless, "
          "but the script will still run on CPU for a correctness check.")

if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | "
          f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =====================================================================
# 0. Real data: enwik9 (byte-level), read only a slice for speed
# =====================================================================

ENWIK9_PATH = Path(__file__).parent / "enwik9"
ENWIK9_BYTES_TO_READ = 20_000_000  # 20MB slice is plenty for this benchmark


def load_enwik9_bytes(path=ENWIK9_PATH, n_bytes=ENWIK9_BYTES_TO_READ):
    if not path.exists():
        raise FileNotFoundError(
            f"Expected enwik9 at {path}. Download it (Hutter Prize dataset) "
            f"and place it at scripts/enwik9, or point ENWIK9_PATH elsewhere."
        )
    with open(path, "rb") as f:
        raw = f.read(n_bytes)
    # byte-level tokenization: vocab_size = 256, no decoding/encoding needed
    data = torch.tensor(list(raw), dtype=torch.uint8)
    print(f"[INFO] Loaded {len(data):,} bytes from {path.name} "
          f"(vocab_size=256, byte-level)")
    return data


# =====================================================================
# 1. Vanilla decoder-only Transformer baseline (standard GPT-style block)
# =====================================================================

class VanillaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, causal_mask):
        n = self.ln1(x)
        out, _ = self.attn(n, n, n, attn_mask=causal_mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class VanillaTransformer(nn.Module):
    """Plain causal decoder-only transformer, O(T^2) full self-attention."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            VanillaBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"seq_len {T} exceeds max_seq_len {self.max_seq_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm_f(x)
        return self.lm_head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_matched_vanilla(target_params, vocab_size, d_model, n_heads, d_ff, max_seq_len, tol=0.05, max_layers=64):
    """Search n_layers so param count is within `tol` (relative) of target_params."""
    best = None
    for n_layers in range(1, max_layers):
        m = VanillaTransformer(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len)
        p = count_params(m)
        diff = abs(p - target_params) / target_params
        if best is None or diff < best[1]:
            best = (n_layers, diff, p)
        if diff <= tol:
            return m, n_layers, p
        del m
    n_layers, diff, p = best
    print(f"[WARN] Could not match within {tol:.0%}, closest: n_layers={n_layers} "
          f"({p:,} params, {diff:.1%} off target)")
    return VanillaTransformer(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len), n_layers, p


# =====================================================================
# 2. VRAM measurement helper (shared by both sweeps)
# =====================================================================

def measure_vram_point(model, idx, vocab_size, device):
    """One forward+backward for an already-built model. Returns peak MB or None on OOM."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        target = torch.randint(0, vocab_size, idx.shape, device=device)
        if isinstance(model, HierarchicalRadialTransformerV7):
            logits = model.lm_head(model(model.tok_emb(idx)))
        else:
            logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
        return peak_mb, "ok"
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        return None, "OOM"


def sweep_seq_len(name, model_fn, seq_lens, vocab_size, device, batch_size=1):
    print(f"\n[{name}] VRAM vs seq_len (batch={batch_size})")
    results = []
    for T in seq_lens:
        try:
            model = model_fn(T).to(device)
            model.train()
            idx = torch.randint(0, vocab_size, (batch_size, T), device=device)
            peak_mb, status = measure_vram_point(model, idx, vocab_size, device)
            if status == "ok":
                print(f"  seq_len={T:7d} | peak_vram={peak_mb:9.1f} MB")
            else:
                print(f"  seq_len={T:7d} | OOM")
            results.append({"seq_len": T, "vram_mb": peak_mb, "status": status})
            del model, idx
        except AssertionError as e:
            print(f"  seq_len={T:7d} | skipped ({e})")
            results.append({"seq_len": T, "vram_mb": None, "status": f"skipped ({e})"})
    return results


def sweep_d_model(name, model_fn, d_models, seq_len, vocab_size, device, batch_size=1):
    print(f"\n[{name}] VRAM vs d_model (seq_len={seq_len}, batch={batch_size})")
    results = []
    for d in d_models:
        try:
            model = model_fn(d).to(device)
            model.train()
            idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            peak_mb, status = measure_vram_point(model, idx, vocab_size, device)
            n_params = count_params(model)
            if status == "ok":
                print(f"  d_model={d:5d} | params={n_params:11,d} | peak_vram={peak_mb:9.1f} MB")
            else:
                print(f"  d_model={d:5d} | params={n_params:11,d} | OOM")
            results.append({"d_model": d, "params": n_params, "vram_mb": peak_mb, "status": status})
            del model, idx
        except AssertionError as e:
            print(f"  d_model={d:5d} | skipped ({e})")
            results.append({"d_model": d, "params": None, "vram_mb": None, "status": f"skipped ({e})"})
    return results


# =====================================================================
# 3. Convergence on real data (enwik9, byte-level)
# =====================================================================

def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).long()
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).long()
    return x.to(device), y.to(device)


def train_and_log(name, model, get_logits_fn, data, seq_len, batch_size, steps, device, log_every=20):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    log = []
    t0 = time.time()
    tokens_seen = 0
    for step in range(1, steps + 1):
        xb, yb = get_batch(data, seq_len, batch_size, device)
        opt.zero_grad(set_to_none=True)
        logits = get_logits_fn(model, xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += batch_size * seq_len

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            log.append({
                "step": step, "loss": loss.item(), "elapsed_s": elapsed,
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-6),
            })
            print(f"  [{name:8s}] step {step:4d}/{steps} | loss {loss.item():.4f} "
                  f"| {elapsed:6.1f}s | {tokens_seen/max(elapsed,1e-6):8.0f} tok/s")
    return log


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    VOCAB = 256          # byte-level
    D_MODEL_REF = 512    # matches HRT ModelConfig defaults
    N_HEADS = 8
    D_FF = 2048

    hrt_cfg_template = dict(
        d_model=D_MODEL_REF, n_outer_latents=128, n_outer_cycles=3, n_outer_heads=N_HEADS,
        n_inner_cycles=6, n_inner_heads=N_HEADS, n_latent_heads=N_HEADS, d_ff=D_FF,
        vocab_size=VOCAB, dropout=0.0, use_compaction=True, causal=True,
    )

    # -----------------------------------------------------------------
    # Benchmark 1: VRAM vs seq_len — directly probes the 64k/128k claim
    # -----------------------------------------------------------------
    print("=" * 70)
    print("BENCHMARK 1: VRAM vs sequence length (d_model=512, batch=1)")
    print("=" * 70)

    seq_lens_hrt = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    seq_lens_vanilla = [1024, 2048, 4096, 8192, 16384]

    def hrt_builder_seq(T):
        cfg = ModelConfig(**{**hrt_cfg_template, "max_seq_len": T})
        return HierarchicalRadialTransformerV7(cfg)

    hrt_vram = sweep_seq_len("HRT", hrt_builder_seq, seq_lens_hrt, VOCAB, DEVICE)

    hrt_ref_params = count_params(HierarchicalRadialTransformerV7(
        ModelConfig(**{**hrt_cfg_template, "max_seq_len": 1024})
    ))
    print(f"\n[Vanilla] matching param count to HRT ({hrt_ref_params:,} params)...")
    _, matched_layers, matched_params = build_matched_vanilla(
        hrt_ref_params, VOCAB, D_MODEL_REF, N_HEADS, D_FF, max_seq_len=1024
    )
    print(f"  -> n_layers={matched_layers} ({matched_params:,} params, "
          f"{abs(matched_params-hrt_ref_params)/hrt_ref_params:.1%} off HRT)")

    def vanilla_builder_seq(T):
        return VanillaTransformer(VOCAB, D_MODEL_REF, matched_layers, N_HEADS, D_FF, max_seq_len=T)

    vanilla_vram = sweep_seq_len("Vanilla", vanilla_builder_seq, seq_lens_vanilla, VOCAB, DEVICE)

    # -----------------------------------------------------------------
    # Benchmark 2: VRAM vs d_model — fixed context, growing model size
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK 2: VRAM vs d_model (seq_len=4096, batch=1)")
    print("=" * 70)

    SEQ_LEN_FIXED = 4096
    d_models = [128, 256, 384, 512, 768, 1024]

    def hrt_builder_dmodel(d):
        heads = max(4, d // 64)  # keep head_dim reasonable as d_model grows
        cfg = ModelConfig(**{**hrt_cfg_template, "d_model": d, "n_outer_heads": heads,
                              "n_inner_heads": heads, "n_latent_heads": heads,
                              "d_ff": d * 4, "max_seq_len": SEQ_LEN_FIXED})
        return HierarchicalRadialTransformerV7(cfg)

    hrt_vram_dmodel = sweep_d_model("HRT", hrt_builder_dmodel, d_models, SEQ_LEN_FIXED, VOCAB, DEVICE)

    def vanilla_builder_dmodel(d):
        heads = max(4, d // 64)
        # match layer count individually per d_model against HRT's param count at that d_model
        target = hrt_vram_dmodel[[x["d_model"] for x in hrt_vram_dmodel].index(d)]["params"]
        _, n_layers, _ = build_matched_vanilla(target, VOCAB, d, heads, d * 4, max_seq_len=SEQ_LEN_FIXED, max_layers=48)
        return VanillaTransformer(VOCAB, d, n_layers, heads, d * 4, max_seq_len=SEQ_LEN_FIXED)

    vanilla_vram_dmodel = sweep_d_model("Vanilla", vanilla_builder_dmodel, d_models, SEQ_LEN_FIXED, VOCAB, DEVICE)

    # -----------------------------------------------------------------
    # Benchmark 3: convergence on REAL text (enwik9, byte-level)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK 3: convergence on enwik9 (byte-level LM)")
    print("=" * 70)

    data = load_enwik9_bytes()

    SEQ_LEN_TRAIN = 512
    BATCH_SIZE = 8
    STEPS = 300

    hrt_cfg = ModelConfig(**{**hrt_cfg_template, "max_seq_len": SEQ_LEN_TRAIN})
    hrt_model = HierarchicalRadialTransformerV7(hrt_cfg)

    def hrt_logits(model, xb):
        return model.lm_head(model(model.tok_emb(xb)))

    print("\n[HRT]")
    hrt_conv = train_and_log("HRT", hrt_model, hrt_logits, data, SEQ_LEN_TRAIN, BATCH_SIZE, STEPS, DEVICE)

    vanilla_model = VanillaTransformer(VOCAB, D_MODEL_REF, matched_layers, N_HEADS, D_FF, max_seq_len=SEQ_LEN_TRAIN)

    def vanilla_logits(model, xb):
        return model(xb)

    print("\n[Vanilla]")
    vanilla_conv = train_and_log("Vanilla", vanilla_model, vanilla_logits, data, SEQ_LEN_TRAIN, BATCH_SIZE, STEPS, DEVICE)

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS (copy these into your plotting script / README)")
    print("=" * 70)

    print("\nhrt_params =", hrt_ref_params)
    print("vanilla_params =", matched_params, f"  # n_layers={matched_layers}")

    print("\nvram_vs_seqlen = {")
    print("    'hrt':", hrt_vram, ",")
    print("    'vanilla':", vanilla_vram, ",")
    print("}")

    print("\nvram_vs_dmodel = {")
    print("    'hrt':", hrt_vram_dmodel, ",")
    print("    'vanilla':", vanilla_vram_dmodel, ",")
    print("}")

    print("\nconvergence_enwik9 = {")
    print("    'hrt':", hrt_conv, ",")
    print("    'vanilla':", vanilla_conv, ",")
    print("}")