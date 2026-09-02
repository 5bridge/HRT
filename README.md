# Hierarchical Radial Transformer (HRT)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Standard self-attention costs `O(T²)` in both compute and memory for a sequence of length `T`. HRT instead compresses the sequence into `K` latent vectors (`K << T`, typically 32–256) and does all reasoning in that latent space. Tokens only ever attend *to* and *from* latents, never to each other directly. This trades some fine-grained token-token modeling for:

- **Sub-quadratic scaling** — attention cost scales with `T·K` instead of `T²`.
- **A fixed-size working memory** — the latent bank is the same size regardless of context length, which is what makes long-context training feasible on small GPUs.
- **Variable compute per input** — an optional implicit-model core (see below) lets the model spend more or fewer reasoning steps depending on the input, rather than a fixed number of layers.

## Architecture overview

HRT processes a sequence in three stages:

```
tokens ──► [Outer: gather] ──► [Inner: reason] ──► [Outer: decode] ──► hidden states
              T → K                 K → K               K → T
```

**1. Outer gather.** A local causal convolution (`LocalConv`, dual dilation + gated blend) first mixes nearby tokens, then a stack of `OuterGatherCycle` layers lets a small set of learned latent vectors (`n_outer_latents`) attend *into* the token sequence and pull out a compressed summary. Position is injected through **HARP** (Hierarchical Adaptive Relational Positioning) — a learned, ALiBi-style distance bias between token positions and latent "budget centers," rather than absolute positional embeddings.

**2. Inner reasoning.** The compressed latents are optionally passed through a **compaction ring** (`RadialCompactionRing`) — a low-rank auto-encoder bottleneck with its own reconstruction+rank loss, trained end-to-end to keep the latent representation compressible. A small set of "center" latents (`n_center_latents`) then run through a **reasoning core** (`InnerGatherCycle`) for a number of cycles. This core can operate in three modes:
   - **Fixed depth** — a plain loop of `n_inner_cycles` iterations (standard).
   - **Internalization / JFB** — trains as an implicit (DEQ-style) model: iterate to near-convergence without gradients, then take one final step with gradients (Jacobi-Free Backpropagation), so backward cost doesn't grow with the number of iterations.
   - **Adaptive halting** — at inference time, iterate until the state stops changing (below `adaptive_epsilon`) or a step budget is hit, so easy inputs finish early and hard inputs get more compute.

**3. Outer decode.** `OuterDecodeCycle` layers let each token position attend back out to the (possibly compressed) latent bank to produce final per-token hidden states. An optional entropy-gated shortcut (`use_ib`) computes a cheap draft logit early and blends it with the full decode output based on prediction entropy — a form of adaptive compute at the output side.

### Long-context memory

Two components extend HRT beyond a single forward pass:

- **`BinaryTemporalTree`** — a binary-counter-style hierarchical memory. Each new chunk's latents are pushed in; when two chunks land in the same tree level, a learned `LatentMerger` (cross-attention + FFN) compresses them into one, freeing the level for the next chunk. This gives geometrically-growing, bounded-size coverage of arbitrarily long history.
- **`GenerationCache` + `TurboQuantizer`** — an incremental-decoding KV cache that rotates keys/values through a fixed random orthogonal matrix before int8-quantizing them (outlier-smoothing, in the spirit of QuaRot/SpinQuant), with online recalibration as new statistics arrive. This is what keeps single-token generation cheap and memory-bounded even at long context.

### Other notable pieces

- **HARP** relative position bias, used at every token↔latent and latent↔latent interaction, with a learned per-head distance slope and a learned latent-latent bias table.
- **QK-norm** (cosine attention with a learned per-head temperature) and **ReZero** residual scaling, both used throughout for training stability without needing warmup tuning.
- **SwiGLU** feed-forward blocks in every attention layer.
- Gradient checkpointing (`use_grad_ckpt`) and `torch.compile` support (`use_compile`) for scaling up when VRAM allows.

## Repository structure

```
hrt/
  config.py          # ModelConfig — all architecture hyperparameters
  model.py            # HierarchicalRadialTransformerV7 — top-level assembly
  ops/                 # qk-norm, SwiGLU FFN
  layers/              # LocalConv, HARP, gather/reason/decode cycles, compaction
  memory/              # LatentMerger, BinaryTemporalTree (long-context memory)
  cache/               # TurboQuantizer, GenerationCache (incremental decoding)
scripts/
  verify_model.py      # forward+backward smoke tests across every config flag
  train.py              # minimal char-level LM training example
```

## Quickstart

```bash
git clone https://github.com/5bridge/HRT.git
cd HRT
pip install -e .

python scripts/verify_model.py   # sanity-check forward/backward across configs
python scripts/train.py          # train a tiny char-level LM (runs on CPU or 6GB+ GPU)
```

Minimal usage:

```python
import torch
from hrt import ModelConfig, HierarchicalRadialTransformerV7

cfg = ModelConfig(
    d_model=256,
    n_outer_latents=32,
    n_outer_cycles=2,
    n_inner_cycles=3,
    vocab_size=5000,
    max_seq_len=2048,
)
model = HierarchicalRadialTransformerV7(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, 128))
x = model.tok_emb(ids)
hidden = model(x)                # (B, T, d_model) — hidden states
logits = model.lm_head(hidden)   # apply the LM head explicitly
```

`forward()` returns hidden states, not logits — this is deliberate, since HRT's primary intended use is as a general feature extractor (`vocab_size=0`) for tasks beyond language modeling (e.g. VAE-style encoders), with an LM head as an optional add-on.

## Configuration

All hyperparameters live in `ModelConfig` (`hrt/config.py`). A few worth knowing about beyond the standard `d_model` / heads / layers:

| Flag | Effect |
|---|---|
| `n_outer_latents` | Size of the compressed latent bank — the main lever for compute/memory vs. quality |
| `use_compaction` | Enables the low-rank compaction ring (extra loss term, `compaction_loss_weight`) |
| `use_internalization` + `use_jfb` | Trains the reasoning core as an implicit/DEQ model instead of fixed depth |
| `use_adaptive_halting` | Variable-depth inference (eval-mode only) |
| `use_ib` | Entropy-gated early-exit shortcut at the decode head |
| `routing_k` | Sparse top-k attention in the reasoning core, for scaling `n_center_latents`/context size |
| `use_q_cache` | Int8 quantized KV cache for incremental generation |
| `cycle_drop` | Randomizes the number of reasoning cycles during training (stochastic depth) |

## Benchmarks

Not yet included in this repository. The claims above (64k context on 6GB VRAM)

## Limitations

- No published head-to-head comparison against standard Transformer baselines yet.
- The latent bottleneck (`n_outer_latents`) trades off fine-grained token-token interaction for efficiency; tasks that need precise pairwise token reasoning (e.g. exact copy/lookup over very long spans) may be harder for this architecture than for full attention.
- Long-context memory (`BinaryTemporalTree`) has not been stress-tested at very large numbers of pushed chunks beyond `btt_max_levels`.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
