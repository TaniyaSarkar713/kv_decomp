# stage1_kv_extract.py
# Stage 1 of the KV-Decomposition handout: extract K/V tensors from a prefill
# pass of Llama-3.2-1B-Instruct, report their dimensions, and measure the
# low-rank decomposition effect (SVD energy spectra, rank-r reconstruction
# error, compression ratio). Plus the handout's Stage-2 check: KL divergence /
# top-1 agreement of the next-token distribution under a rank-r cache.
#
# Follows: KV-Decomposition_Beginner-Handout (Part V) and the LLaMA-3.2 local
# inference guide (Section 6.2). transformers 4.57.x cache API: cache[i] ->
# (K, V); write back via cache.layers[i].keys / .values.

import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

NAME = "unsloth/Llama-3.2-1B-Instruct"  # ungated mirror of meta-llama/Llama-3.2-1B-Instruct
DEVICE = "cuda:0"

assert torch.cuda.is_available(), "CUDA GPU required"
print(f"device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"torch {torch.__version__}")

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, dtype=torch.bfloat16, device_map=DEVICE
).eval()

cfg = model.config
print(
    f"model: {NAME}\n"
    f"  layers={cfg.num_hidden_layers}  hidden={cfg.hidden_size}  "
    f"q_heads={cfg.num_attention_heads}  kv_heads={cfg.num_key_value_heads}  "
    f"head_dim={cfg.head_dim}"
)

# Varied natural text (NOT a repeated sentence -- repetition would artificially
# inflate low-rankness of the cache and bias the spectrum measurement).
PROMPT = """The theory of general relativity describes gravity not as a force
but as the curvature of spacetime caused by mass and energy. Light passing near
a massive object follows this curvature, which is why distant galaxies can act
as gravitational lenses, bending and magnifying the images of objects behind
them. Precise clocks on satellites must be corrected for both special and
general relativistic effects, or GPS positions would drift by kilometers each
day. In a very different domain, the process of fermentation has shaped human
cuisine for thousands of years. Bread rises because yeast metabolizes sugars
and releases carbon dioxide, while sourdough cultures pair wild yeasts with
lactic acid bacteria that give the loaf its characteristic tang. Cheese, soy
sauce, kimchi, and chocolate all depend on carefully managed microbial
communities, and the flavors we associate with each food come from specific
metabolic byproducts. The history of navigation offers another thread. Before
satellite positioning, sailors fixed their position with a sextant, an accurate
clock, and printed tables of celestial coordinates. The invention of the marine
chronometer by John Harrison in the eighteenth century finally made longitude
measurable at sea, transforming trade routes and naval strategy. Modern inertial
navigation systems still carry forward the same principle: integrate careful
measurements over time to estimate where you are. In computing, a hash table
stores key-value pairs by mapping each key through a hash function to a bucket
index, giving constant-time lookups on average. When two keys collide, the
table either chains entries in a linked list or probes for the next open slot.
The choice of load factor balances memory against speed, and resizing the table
requires rehashing every stored key. Compilers, databases, caches, and
interpreters all lean on this one data structure, and its behavior under
adversarial inputs motivated the development of randomized hashing. Finally,
consider the water cycle: evaporation lifts moisture from oceans, condensation
forms clouds, precipitation returns water to the surface, and rivers carry it
back to the sea, redistributing heat around the planet as they do."""

inputs = tok(PROMPT, return_tensors="pt").to(model.device)
S_prompt = inputs.input_ids.shape[1]
print(f"\nprompt tokens: {S_prompt}")

# ---------------- Stage 1a: extract the KV cache ----------------
with torch.no_grad():
    out = model(**inputs, use_cache=True)  # one prefill pass
cache = out.past_key_values
print(f"\ncache type: {type(cache).__name__}, layers: {len(cache)}")

K0, V0 = cache[0]
print(f"per-layer K shape: {tuple(K0.shape)}  dtype={K0.dtype}")
print(f"per-layer V shape: {tuple(V0.shape)}  dtype={V0.dtype}")
B, H, S, D = K0.shape

# sanity: all layers same shape
for i in range(len(cache)):
    Ki, Vi = cache[i]
    assert Ki.shape == K0.shape and Vi.shape == K0.shape

bytes_per_tok = 2 * len(cache) * H * D * K0.element_size()
total_bytes = sum(cache[i][0].numel() * cache[i][0].element_size() * 2 for i in range(len(cache)))
print(f"KV bytes/token (analytic): {bytes_per_tok} = {bytes_per_tok/1024:.0f} KiB")
print(f"total KV cache for this prompt: {total_bytes/2**20:.1f} MiB "
      f"({S} tokens x {bytes_per_tok/1024:.0f} KiB)")

# ---------------- Stage 1b: singular-value spectra ----------------
def energy_vs_rank(mat):  # mat: [S, d] -> cum[r-1] = energy kept at rank r
    s = torch.linalg.svdvals(mat.float())
    return (torch.cumsum(s**2, 0) / (s**2).sum()).cpu()

def rank_for_energy(cum, frac):
    return int((cum < frac).sum().item()) + 1

# energy curves averaged over all KV heads, per layer
layers = list(range(len(cache)))
k_curves, v_curves = [], []
for i in layers:
    K, V = cache[i]
    k_curves.append(torch.stack([energy_vs_rank(K[0, h]) for h in range(H)]).mean(0))
    v_curves.append(torch.stack([energy_vs_rank(V[0, h]) for h in range(H)]).mean(0))

print("\nrank needed to retain 90% / 95% / 99% of energy "
      f"(head-averaged, post-RoPE keys, head_dim={D}):")
print(f"{'layer':>5} | {'K@90':>5} {'K@95':>5} {'K@99':>5} | {'V@90':>5} {'V@95':>5} {'V@99':>5}")
summary = []
for i in layers:
    kr = [rank_for_energy(k_curves[i], f) for f in (0.90, 0.95, 0.99)]
    vr = [rank_for_energy(v_curves[i], f) for f in (0.90, 0.95, 0.99)]
    summary.append({"layer": i, "K": kr, "V": vr})
    print(f"{i:>5} | {kr[0]:>5} {kr[1]:>5} {kr[2]:>5} | {vr[0]:>5} {vr[1]:>5} {vr[2]:>5}")

# ---------------- Stage 1c: rank-r reconstruction error & compression ----------------
def svd_compress(x, r):
    """x: [B, H, S, D] -> rank-r reconstruction of each [S, D] slice."""
    b, h, n, d = x.shape
    flat = x.float().reshape(b * h, n, d)
    U, s, Vh = torch.linalg.svd(flat, full_matrices=False)
    r = min(r, s.shape[-1])
    recon = (U[..., :r] * s[..., :r].unsqueeze(-2)) @ Vh[..., :r, :]
    return recon.reshape(b, h, n, d)

RANKS = [4, 8, 16, 32]
print("\nrank-r reconstruction: mean relative Frobenius error over layers "
      "(and per-head compression ratio S*D / r*(S+D)):")
print(f"{'rank':>4} | {'K err':>8} {'V err':>8} | {'compression':>11}")
err_table = {}
for r in RANKS:
    k_errs, v_errs = [], []
    for i in layers:
        K, V = cache[i]
        k_errs.append((torch.norm(K.float() - svd_compress(K, r)) / torch.norm(K.float())).item())
        v_errs.append((torch.norm(V.float() - svd_compress(V, r)) / torch.norm(V.float())).item())
    ratio = (S * D) / (r * (S + D))
    err_table[r] = {"K": k_errs, "V": v_errs, "ratio": ratio}
    print(f"{r:>4} | {np.mean(k_errs):>8.4f} {np.mean(v_errs):>8.4f} | {ratio:>10.2f}x")

# ---------------- Stage 2: effect on the next-token distribution ----------------
def low_rank_reconstruct(x, r):
    return svd_compress(x, r).to(x.dtype)

class LowRankCache(DynamicCache):
    def __init__(self, rank=32, **kw):
        super().__init__(**kw)
        self.rank = rank
        self._done = set()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx not in self._done and keys.shape[-2] > 1:  # once, after prefill
            keys = low_rank_reconstruct(keys, self.rank)
            values = low_rank_reconstruct(values, self.rank)
            self.layers[layer_idx].keys = keys
            self.layers[layer_idx].values = values
            self._done.add(layer_idx)
        return keys, values

def next_logits(cache_ctor):
    c = cache_ctor()
    with torch.no_grad():
        o = model(**inputs, past_key_values=c, use_cache=True)
    return o.logits[0, -1].float()

full = next_logits(lambda: DynamicCache(config=model.config))
print("\nStage 2 -- next-token distribution, full cache vs rank-r cache (K and V compressed):")
print(f"{'rank':>4} | {'KL(full||lr)':>12} {'top1_agree':>10}")
stage2 = []
for r in RANKS + [64]:
    lr = next_logits(lambda: LowRankCache(rank=r, config=model.config))
    kl = torch.nn.functional.kl_div(
        lr.log_softmax(-1), full.softmax(-1), reduction="sum"
    ).item()
    agree = bool((full.argmax() == lr.argmax()).item())
    stage2.append({"rank": r, "kl": kl, "top1_agree": agree})
    print(f"{r:>4} | {kl:>12.4f} {str(agree):>10}")

# ---------------- plots ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
show = [0, len(cache) // 2, len(cache) - 1]
colors = ["#283593", "#2E7D32", "#B23A00"]
for c, i in zip(colors, show):
    axes[0].plot(range(1, D + 1), k_curves[i], color=c, label=f"layer {i} K")
    axes[0].plot(range(1, D + 1), v_curves[i], color=c, ls="--", label=f"layer {i} V")
axes[0].axhline(0.95, color="gray", ls=":")
axes[0].set_xlabel("rank r"); axes[0].set_ylabel("energy retained")
axes[0].set_title(f"Llama-3.2-1B KV energy vs rank (post-RoPE, S={S})")
axes[0].legend(fontsize=8)

k90 = [s["K"][1] for s in summary]; v90 = [s["V"][1] for s in summary]
axes[1].plot(layers, k90, "o-", color="#283593", label="K rank@95% energy")
axes[1].plot(layers, v90, "s--", color="#B23A00", label="V rank@95% energy")
axes[1].axhline(D, color="gray", ls=":", label=f"full rank ({D})")
axes[1].set_xlabel("layer"); axes[1].set_ylabel("rank")
axes[1].set_title("Rank needed for 95% energy, by depth")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/stage1_spectrum.png", dpi=150)
print("\nsaved plot: results/stage1_spectrum.png")

with open("results/stage1_results.json", "w") as f:
    json.dump(
        {
            "model": NAME,
            "device": torch.cuda.get_device_name(0),
            "kv_shape_per_layer": list(K0.shape),
            "layers": len(cache),
            "kv_bytes_per_token": bytes_per_tok,
            "energy_rank_summary": summary,
            "reconstruction": {
                str(r): {
                    "K_err_mean": float(np.mean(v["K"])),
                    "V_err_mean": float(np.mean(v["V"])),
                    "compression": v["ratio"],
                }
                for r, v in err_table.items()
            },
            "stage2_next_token": stage2,
        },
        f,
        indent=2,
    )
print("saved results: results/stage1_results.json")
