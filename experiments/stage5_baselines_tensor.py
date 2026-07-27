# stage5_baselines_tensor.py
# Stage 5 of the KV-Decomposition handout:
#  (a) KIVI-style 2-bit quantization baseline, placed on the same
#      KL-vs-compression axes as the Stage-4 SVD sweep;
#  (b) the tensor step-up: per-layer Tucker and Tensor-Train decompositions of
#      the [heads x tokens x head_dim] tensor (tensorly), which share structure
#      ACROSS the 8 KV heads -- attacking the 64/r ceiling of per-head SVD.
# Accuracy protocol identical to Stage 4 (next-token KL vs full cache).

import json
import time
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker, tensor_train
from transformers import DynamicCache
from kv_common import load_model, long_text_ids, svd_compress, kivi_quant_dequant

tl.set_backend("pytorch")
torch.manual_seed(0)
tok, model = load_model()
cfg = model.config
H, D = cfg.num_key_value_heads, cfg.head_dim
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
      f"tensorly {tl.__version__}")

KIVI_CONTEXTS = [512, 2048, 8192]
TENSOR_CTX = 2048
SVD_RANKS = [4, 8, 16, 32, 48]
# (r_heads, r_tokens, r_dim): r_heads < 8 shares a subspace across heads
TUCKER_CFGS = [(8, 256, 32), (8, 128, 16), (4, 64, 16), (8, 64, 8), (4, 32, 8)]
TT_CFGS = [(8, 32), (8, 16), (4, 16), (8, 8), (4, 8)]  # (R1, R2) for [H,S,D]

results = {"kivi": [], "svd": [], "tucker": [], "tt": []}


def prefill(ctx):
    ids = long_text_ids(tok, ctx, model.device)
    prefill_ids, last = ids[:, :-1], ids[:, -1:]
    c = DynamicCache(config=cfg)
    with torch.no_grad():
        model(input_ids=prefill_ids, past_key_values=c, use_cache=True)
    return [(c[i][0], c[i][1]) for i in range(len(c))], last


def logits_with(tensors, last):
    c = DynamicCache(config=cfg)
    for i, (K, V) in enumerate(tensors):
        c.update(K, V, i)
    with torch.no_grad():
        o = model(input_ids=last, past_key_values=c, use_cache=True)
    return o.logits[0, -1].float()


def kl_top1(full, lr):
    kl = torch.nn.functional.kl_div(
        lr.log_softmax(-1), full.softmax(-1), reduction="sum"
    ).item()
    return kl, bool((full.argmax() == lr.argmax()).item())


# ---------------- (a) KIVI 2-bit baseline ----------------
print("\n--- KIVI-style 2-bit (keys per-channel, values per-token, "
      "group 32, residual 32) ---")
for ctx in KIVI_CONTEXTS:
    base, last = prefill(ctx)
    full = logits_with(base, last)
    qt = []
    ratio = None
    for K, V in base:
        Kq, Vq, ratio = kivi_quant_dequant(K, V)
        qt.append((Kq, Vq))
    kl, agree = kl_top1(full, logits_with(qt, last))
    rel_k = np.mean([(torch.norm(K.float() - Kq.float()) / torch.norm(K.float())).item()
                     for (K, V), (Kq, Vq) in zip(base, qt)])
    rel_v = np.mean([(torch.norm(V.float() - Vq.float()) / torch.norm(V.float())).item()
                     for (K, V), (Kq, Vq) in zip(base, qt)])
    results["kivi"].append({"ctx": ctx, "kl": kl, "top1_agree": agree,
                            "compression": ratio, "rel_err_K": rel_k, "rel_err_V": rel_v})
    print(f"ctx {ctx:>5}: KL {kl:.4f}  top1 {agree}  compression {ratio:.2f}x  "
          f"relerr K {rel_k:.4f} V {rel_v:.4f}")
    del base, qt
    torch.cuda.empty_cache()

# ---------------- (b) tensor decompositions at ctx 2048 ----------------
base, last = prefill(TENSOR_CTX)
full = logits_with(base, last)
S = base[0][0].shape[-2]
dense_elems = H * S * D
print(f"\n--- tensor decompositions, ctx {TENSOR_CTX} (S={S}), per-layer "
      f"[{H} x {S} x {D}] ---")

# per-head SVD reference on the SAME base (rel err here; KL matches stage 4)
for r in SVD_RANKS:
    ratio = (S * D) / (r * (S + D))
    ek, ev, comp = [], [], []
    for K, V in base:
        Kh, Vh = svd_compress(K, r), svd_compress(V, r)
        ek.append((torch.norm(K.float() - Kh) / torch.norm(K.float())).item())
        ev.append((torch.norm(V.float() - Vh) / torch.norm(V.float())).item())
        comp.append((Kh.to(K.dtype), Vh.to(V.dtype)))
    kl, agree = kl_top1(full, logits_with(comp, last))
    results["svd"].append({"rank": r, "compression": ratio, "kl": kl,
                           "top1_agree": agree,
                           "rel_err_K": float(np.mean(ek)), "rel_err_V": float(np.mean(ev))})
    print(f"SVD r={r:<3}          compression {ratio:5.2f}x  KL {kl:8.4f}  "
          f"top1 {agree}  relerr K {np.mean(ek):.4f} V {np.mean(ev):.4f}")
    del comp
    torch.cuda.empty_cache()


def run_family(name, cfgs, decompose, n_elems):
    for c in cfgs:
        t0 = time.perf_counter()
        ratio = dense_elems / n_elems(c)
        ek, ev, comp = [], [], []
        for K, V in base:
            outs = []
            for X, errs in ((K, ek), (V, ev)):
                T = X[0].float()  # [H, S, D]
                Xh = decompose(T, c)
                errs.append((torch.norm(T - Xh) / torch.norm(T)).item())
                outs.append(Xh.unsqueeze(0).to(X.dtype))
            comp.append((outs[0], outs[1]))
        kl, agree = kl_top1(full, logits_with(comp, last))
        secs = time.perf_counter() - t0
        results[name].append({"cfg": list(c), "compression": ratio, "kl": kl,
                              "top1_agree": agree, "rel_err_K": float(np.mean(ek)),
                              "rel_err_V": float(np.mean(ev)), "seconds": secs})
        print(f"{name} {str(c):<14} compression {ratio:5.2f}x  KL {kl:8.4f}  "
              f"top1 {agree}  relerr K {np.mean(ek):.4f} V {np.mean(ev):.4f}  "
              f"({secs:.0f}s)")
        del comp
        torch.cuda.empty_cache()


run_family(
    "tucker", TUCKER_CFGS,
    lambda T, c: tl.tucker_to_tensor(tucker(T, rank=list(c), init="svd", n_iter_max=20)),
    lambda c: c[0] * c[1] * c[2] + H * c[0] + S * c[1] + D * c[2],
)
run_family(
    "tt", TT_CFGS,
    lambda T, c: tl.tt_to_tensor(tensor_train(T, rank=[1, c[0], c[1], 1])),
    lambda c: H * c[0] + c[0] * S * c[1] + c[1] * D,
)

with open("results/stage5_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved: results/stage5_results.json")

# ---------------- plot: methods on the same axes ----------------
COLOR = {"svd": "#5C6BC0", "tucker": "#2E7D32", "tt": "#B23A00", "kivi": "#616161"}
MARK = {"svd": "^", "tucker": "o", "tt": "s", "kivi": "D"}
LABEL = {"svd": "per-head SVD", "tucker": "Tucker [H,S,D]",
         "tt": "Tensor-Train [H,S,D]", "kivi": "KIVI-style 2-bit"}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax = axes[0]
for m in ("svd", "tucker", "tt"):
    pts = sorted(results[m], key=lambda a: a["compression"])
    ax.plot([p["compression"] for p in pts], [max(p["kl"], 1e-4) for p in pts],
            color=COLOR[m], lw=1.2, alpha=0.8, label=LABEL[m])
    for p in pts:
        ax.plot(p["compression"], max(p["kl"], 1e-4), MARK[m], color=COLOR[m],
                ms=6, markerfacecolor=COLOR[m] if p["top1_agree"] else "white")
kp = next(p for p in results["kivi"] if p["ctx"] == TENSOR_CTX)
ax.plot(kp["compression"], max(kp["kl"], 1e-4), MARK["kivi"], color=COLOR["kivi"],
        ms=8, markerfacecolor=COLOR["kivi"] if kp["top1_agree"] else "white",
        label=LABEL["kivi"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("compression ratio (K+V cache)")
ax.set_ylabel("KL(full || compressed)")
ax.set_title(f"Method comparison, ctx {TENSOR_CTX} (filled = top-1 preserved)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
for m in ("svd", "tucker", "tt"):
    pts = sorted(results[m], key=lambda a: a["compression"])
    xs = [p["compression"] for p in pts]
    ax.plot(xs, [p["rel_err_K"] for p in pts], "-", marker=MARK[m], ms=5,
            color=COLOR[m], lw=1.2, label=f"{LABEL[m]} K")
    ax.plot(xs, [p["rel_err_V"] for p in pts], "--", marker=MARK[m], ms=5,
            color=COLOR[m], lw=1.2, alpha=0.6, label=f"{LABEL[m]} V")
ax.set_xscale("log")
ax.set_xlabel("compression ratio (K+V cache)")
ax.set_ylabel("relative Frobenius error")
ax.set_title("Reconstruction error (K solid, V dashed)")
ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig("results/stage5_comparison.png", dpi=150)
print("saved: results/stage5_comparison.png")
