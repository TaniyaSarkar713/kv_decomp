# stage4_sweep.py
# Stage 4 of the KV-Decomposition handout: sweep rank x context length,
# separately for keys-only / values-only / both, and produce the
# compression-vs-accuracy-vs-latency Pareto data + money plot.
#
# Accuracy protocol (extends Stage 2): prefill the first S-1 WikiText-2 tokens
# with a full cache, then feed the last token once with (a) the full cache and
# (b) a rank-r compressed copy, and compare next-token distributions
# (KL divergence, top-1 agreement). Same code path for both, greedy semantics.
# Latency protocol: TPOT of the factored (memory-honest) cache vs dense.

import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import DynamicCache
from kv_common import (
    load_model, long_text_ids, svd_compress, FactoredLowRankCache, time_ms,
)

torch.manual_seed(0)
tok, model = load_model()
cfg = model.config
D = cfg.head_dim
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

CONTEXTS = [512, 2048, 8192]
RANKS = [4, 8, 16, 32, 48]
MODES = [("K", True, False), ("V", False, True), ("KV", True, True)]
N_NEW = 32  # decode length for the latency part

results = {"accuracy": [], "latency": []}


def compression_ratio(S, r, ck, cv):
    """Overall prefill-cache compression (K+V), per head: dense 2*S*D elems."""
    dense = 2 * S * D
    comp = (r * (S + D) if ck else S * D) + (r * (S + D) if cv else S * D)
    return dense / comp


# ---------------- accuracy sweep ----------------
for ctx in CONTEXTS:
    ids = long_text_ids(tok, ctx, model.device)
    prefill_ids, last = ids[:, :-1], ids[:, -1:]
    S = prefill_ids.shape[1]

    base = DynamicCache(config=cfg)
    with torch.no_grad():
        model(input_ids=prefill_ids, past_key_values=base, use_cache=True)
    base_tensors = [(base[i][0], base[i][1]) for i in range(len(base))]
    del base

    def logits_with(tensors):
        c = DynamicCache(config=cfg)
        for i, (K, V) in enumerate(tensors):
            c.update(K, V, i)
        with torch.no_grad():
            o = model(input_ids=last, past_key_values=c, use_cache=True)
        return o.logits[0, -1].float()

    full = logits_with(base_tensors)
    sanity = torch.nn.functional.kl_div(
        logits_with(base_tensors).log_softmax(-1), full.softmax(-1), reduction="sum"
    ).item()
    print(f"\n=== context {ctx} (prefill S={S}) ===  sanity KL(full||full)={sanity:.2e}")
    print(f"{'mode':>4} {'rank':>4} | {'KL':>9} | {'top1':>5} | {'compress':>8}")

    for mode, ck, cv in MODES:
        for r in RANKS:
            tensors = [
                (
                    svd_compress(K, r).to(K.dtype) if ck else K,
                    svd_compress(V, r).to(V.dtype) if cv else V,
                )
                for K, V in base_tensors
            ]
            lr = logits_with(tensors)
            kl = torch.nn.functional.kl_div(
                lr.log_softmax(-1), full.softmax(-1), reduction="sum"
            ).item()
            agree = bool((full.argmax() == lr.argmax()).item())
            ratio = compression_ratio(S, r, ck, cv)
            results["accuracy"].append(
                {"ctx": ctx, "mode": mode, "rank": r, "kl": kl,
                 "top1_agree": agree, "compression": ratio}
            )
            print(f"{mode:>4} {r:>4} | {kl:>9.4f} | {str(agree):>5} | {ratio:>7.2f}x")
    del base_tensors
    torch.cuda.empty_cache()

# ---------------- latency sweep (factored cache, K and V compressed) ----------------
print("\n--- latency: factored cache TPOT vs dense ---")
for ctx in CONTEXTS:
    ids = long_text_ids(tok, ctx, model.device)

    def gen(ctor, n):
        return model.generate(
            input_ids=ids, do_sample=False, max_new_tokens=n, min_new_tokens=n,
            past_key_values=ctor(), use_cache=True, pad_token_id=tok.eos_token_id,
        )

    def tpot_of(ctor):
        ttft = time_ms(lambda: gen(ctor, 1), warmup=1, iters=3)
        e2e = time_ms(lambda: gen(ctor, N_NEW), warmup=1, iters=3)
        return (e2e["median"] - ttft["median"]) / (N_NEW - 1), ttft["median"]

    dense_tpot, dense_ttft = tpot_of(lambda: DynamicCache(config=cfg))
    results["latency"].append(
        {"ctx": ctx, "variant": "dense", "tpot_ms": dense_tpot, "ttft_ms": dense_ttft}
    )
    print(f"ctx {ctx:>5}  dense        TPOT {dense_tpot:6.2f} ms  TTFT {dense_ttft:8.1f} ms")
    for r in RANKS:
        tpot, ttft = tpot_of(lambda: FactoredLowRankCache(rank=r, config=cfg))
        results["latency"].append(
            {"ctx": ctx, "variant": f"factored-r{r}", "rank": r,
             "tpot_ms": tpot, "ttft_ms": ttft,
             "tpot_vs_dense": tpot / dense_tpot,
             "compression": compression_ratio(ctx, r, True, True)}
        )
        print(f"ctx {ctx:>5}  factored-r{r:<3} TPOT {tpot:6.2f} ms  TTFT {ttft:8.1f} ms  "
              f"({100 * tpot / dense_tpot:.0f}% of dense)")

with open("results/stage4_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved: results/stage4_results.json")

# ---------------- money plot ----------------
CTX_COLOR = {512: "#9FA8DA", 2048: "#5C6BC0", 8192: "#283593"}  # one hue, light->dark
MODE_MARK = {"K": "o", "V": "s", "KV": "^"}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax = axes[0]
for ctx in CONTEXTS:
    for mode, _, _ in MODES:
        pts = [a for a in results["accuracy"] if a["ctx"] == ctx and a["mode"] == mode]
        pts.sort(key=lambda a: a["compression"])
        xs = [p["compression"] for p in pts]
        ys = [max(p["kl"], 1e-4) for p in pts]
        ax.plot(xs, ys, color=CTX_COLOR[ctx], lw=1.2, alpha=0.8,
                ls={"K": "-", "V": ":", "KV": "--"}[mode])
        for p in pts:
            filled = p["top1_agree"]
            ax.plot(p["compression"], max(p["kl"], 1e-4), MODE_MARK[mode],
                    color=CTX_COLOR[ctx], ms=6,
                    markerfacecolor=CTX_COLOR[ctx] if filled else "white")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("compression ratio (K+V cache)")
ax.set_ylabel("KL(full || compressed) of next-token dist.")
ax.set_title("Accuracy vs compression (filled = top-1 token preserved)")
ax.grid(alpha=0.3)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=CTX_COLOR[c], lw=2, label=f"ctx {c}") for c in CONTEXTS]
handles += [Line2D([], [], color="gray", marker=MODE_MARK[m], ls="", label=f"{m} compressed")
            for m, _, _ in MODES]
ax.legend(handles=handles, fontsize=8, ncol=2)

ax = axes[1]
for ctx in CONTEXTS:
    pts = [l for l in results["latency"] if l["ctx"] == ctx and l["variant"] != "dense"]
    pts.sort(key=lambda a: a["compression"])
    ax.plot([p["compression"] for p in pts],
            [100 * p["tpot_vs_dense"] for p in pts],
            "o-", color=CTX_COLOR[ctx], lw=1.2, label=f"ctx {ctx}")
ax.axhline(100, color="gray", ls="--", lw=1)
ax.text(ax.get_xlim()[1], 101, "dense baseline", ha="right", va="bottom",
        fontsize=8, color="gray")
ax.set_xscale("log")
ax.set_xlabel("compression ratio (K+V cache)")
ax.set_ylabel("TPOT, % of dense")
ax.set_title("Decode latency of factored cache vs compression")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("results/stage4_pareto.png", dpi=150)
print("saved: results/stage4_pareto.png")
