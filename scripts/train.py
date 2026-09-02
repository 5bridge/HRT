import time
import torch
import torch.nn.functional as F
from hrt import ModelConfig, HierarchicalRadialTransformerV7

# ---------------------------------------------------------------
# 1. Toy dataset — repeated text, char-level tokenization.
#    Replace `TEXT` with your own corpus for a real run.
# ---------------------------------------------------------------
TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "hierarchical radial transformers compress long context into latents. "
) * 200

chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

SEQ_LEN = 128


def get_batch(batch_size, device):
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (batch_size,))
    x = torch.stack([data[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + SEQ_LEN + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------
# 2. Small config — tuned to fit comfortably on ~6GB VRAM.
# ---------------------------------------------------------------
cfg = ModelConfig(
    d_model=256,
    n_outer_latents=32,
    n_outer_cycles=2,
    n_outer_heads=4,
    n_inner_cycles=3,
    n_inner_heads=4,
    n_latent_heads=4,
    d_ff=512,
    max_seq_len=SEQ_LEN,
    vocab_size=vocab_size,
    dropout=0.0,
    use_compaction=True,
    use_grad_ckpt=False,   # set True if you push d_model/seq_len higher and hit OOM
    causal=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HierarchicalRadialTransformerV7(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,} | device: {device} | vocab_size: {vocab_size}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

STEPS = 300
BATCH_SIZE = 16
LOG_EVERY = 20

model.train()
t0 = time.time()
for step in range(1, STEPS + 1):
    xb, yb = get_batch(BATCH_SIZE, device)
    x_emb = model.tok_emb(xb)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        hidden = model(x_emb)              # (B, T, D) — hidden states, not logits
        logits = model.lm_head(hidden)     # (B, T, vocab) — apply head manually
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    if step % LOG_EVERY == 0 or step == 1:
        elapsed = time.time() - t0
        mem = f"{torch.cuda.memory_allocated() / 1e6:.0f}MB" if device == "cuda" else "n/a"
        print(f"step {step:4d}/{STEPS} | loss {loss.item():.4f} | {elapsed:.1f}s | vram {mem}")

# ---------------------------------------------------------------
# 3. Quick qualitative check — generate a short continuation.
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    prompt = "the quick"
    ids = torch.tensor([[stoi[c] for c in prompt]], device=device)
    prompt_emb = model.tok_emb(ids)
    out_embs = model.generate(prompt_emb, max_new=60, temperature=0.8, top_k=10)
    # decode by nearest embedding match back to ids is lossy for raw embeddings;
    # instead re-run through lm_head on hidden states for a clean sample:
    hidden = model(out_embs)
    logits = model.lm_head(hidden)
    sampled_ids = logits.argmax(dim=-1)[0].tolist()
    print("\nSample:", "".join(itos[i] for i in sampled_ids))

torch.save(model.state_dict(), "hrt_toy_checkpoint.pt")
print("\nSaved checkpoint to hrt_toy_checkpoint.pt")