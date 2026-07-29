# stage8_packed_frontier.py
# Stage 8: the physical packed baseline frontier + asymmetric K/V grid
# (Experiments 1+2 of RESULTS_AND_NEXT_EXPERIMENTS.md, merged).
#
# Everything before this stage accounted low-bit storage ANALYTICALLY.
# Here quantized old blocks are actually packed (uint8, 2 or 4 values/byte,
# fp16 group stats counted) via kv_hybrid.pack_quant, and every config's
# bytes/token/layer is measured from the stored tensors. Latent configs count
# coefficient + residual value/index bytes (fixed bases are weight-side and
# amortized across all prompts, reported in meta).
#
# Grid hypotheses (doc conclusions 2 & 5):
#   (a) asymmetric K-latent + V-int4 Pareto-dominates symmetric latent KV;
#   (b) latent + SAGE influence residual is competitive with packed KIVI-style
#       int2/int4 at equal physical bytes -- if not, quantization wins and the
#       tensor line narrows to capacity-only.
#
# Protocol: stage-6/7 hybrid path -- dense prefill, hot window W=64 exact,
# old block re-coded once; teacher-forced 256-token held-out tail gives
# mean KL vs dense AND true-token perplexity in the same forward.
# quantization layout follows KIVI: K per-channel (groups along tokens),
# V per-token (groups along channels).

import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from kv_common import load_model
from kv_hybrid import Hybrid, pack_quant, unpack_quant

torch.manual_seed(0)
tok, model = load_model()
hy = Hybrid(model)
H, D, P, L = hy.H, hy.D, hy.P, hy.L
DEV = model.device
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

W = 64
T_EVAL = 256
CONTEXTS = [2048, 8192]
OFFSETS = [0, 40000, 80000]
GROUP = 32

# ---------------- data ----------------
_ids_cache = {}


def window_ids(offset, n):
    if "test" not in _ids_cache:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        _ids_cache["test"] = tok("\n\n".join(data["text"]),
                                 return_tensors="pt").input_ids
    ids = _ids_cache["test"]
    assert ids.shape[1] >= offset + n
    if offset == 0:
        return ids[:, :n].to(DEV)
    bos = torch.full((1, 1), tok.bos_token_id, dtype=ids.dtype)
    return torch.cat([bos, ids[:, offset + 1: offset + n]], dim=1).to(DEV)


# ---------------- codecs ----------------
def code_tensor(X, li, ti, codec):
    """One old-block tensor [H, S, D] fp32 through a codec round trip.
    Returns (Xhat, physical_bytes). codec: ("native",) | ("lat", r) |
    ("int", bits)."""
    S = X.shape[1]
    if codec[0] == "native":
        return X, H * S * D * 2
    if codec[0] == "lat":
        r = codec[1]
        return hy.project_fixed(X, li, ti, r), H * S * r * 2
    if codec[0] == "int":
        bits = codec[1]
        grp_dim = 1 if ti == 0 else 2      # K per-channel, V per-token (KIVI)
        packed, mn, sc, nbytes, meta = pack_quant(X, GROUP, grp_dim, bits=bits)
        return unpack_quant(packed, mn, sc, meta, GROUP, bits=bits), nbytes
    raise ValueError(codec)


def compress_asym(base, kcodec, vcodec, residual, abar):
    """residual: None | ("K"|"KV", rho) -- influence rule, applied to the
    latent-coded tensors only. Returns (tensors, bytes_per_tok_layer)."""
    out, total_bytes, S_old = [], 0, None
    for li, (K, V) in enumerate(base):
        S = K.shape[-2]
        S_old = S - W
        Kf, Vf = K[0, :, :S_old].float(), V[0, :, :S_old].float()
        Kh, bK = code_tensor(Kf, li, 0, kcodec)
        Vh, bV = code_tensor(Vf, li, 1, vcodec)
        rbytes = 0
        if residual is not None:
            which, rho = residual
            eK_full = Kf - Kh if kcodec[0] == "lat" else torch.zeros_like(Kf)
            eV_full = (Vf - Vh if (vcodec[0] == "lat" and which == "KV")
                       else torch.zeros_like(Vf))
            eK, eV = hy.residual(eK_full, eV_full, li, "influence", rho, abar)
            if kcodec[0] == "lat":
                Kh = Kh + eK
                rbytes += rho * H * S_old * P * (2 * 2 + 4)
            if vcodec[0] == "lat" and which == "KV":
                Vh = Vh + eV
                rbytes += rho * H * S_old * P * (2 * 2 + 4)
        total_bytes += bK + bV + rbytes
        Kc = torch.cat([Kh.unsqueeze(0).to(K.dtype), K[:, :, S_old:]], dim=-2)
        Vc = torch.cat([Vh.unsqueeze(0).to(V.dtype), V[:, :, S_old:]], dim=-2)
        out.append((Kc, Vc))
    return out, total_bytes / (S_old * L)


CONFIGS = [
    # name, K codec, V codec, residual, family
    ("dense", ("native",), ("native",), None, "ref"),
    ("Kint2/Vint2", ("int", 2), ("int", 2), None, "quant"),
    ("Kint4/Vint4", ("int", 4), ("int", 4), None, "quant"),
    ("Kint4/Vnat", ("int", 4), ("native",), None, "oneside"),
    ("Knat/Vint4", ("native",), ("int", 4), None, "oneside"),
    ("Klat16/Vlat16", ("lat", 16), ("lat", 16), None, "symlat"),
    ("Klat16/Vlat16+inf5", ("lat", 16), ("lat", 16), ("KV", 0.05), "symlat"),
    ("Klat32/Vint4", ("lat", 32), ("int", 4), None, "asym"),
    ("Klat32/Vint4+infK5", ("lat", 32), ("int", 4), ("K", 0.05), "asym"),
    ("Klat16/Vint4", ("lat", 16), ("int", 4), None, "asym"),
    ("Klat16/Vint4+infK5", ("lat", 16), ("int", 4), ("K", 0.05), "asym"),
    ("Klat32/Vlat8+inf5", ("lat", 32), ("lat", 8), ("KV", 0.05), "vlow"),
]

results = {"meta": {"W": W, "t_eval": T_EVAL, "contexts": CONTEXTS,
                    "offsets": OFFSETS, "group": GROUP,
                    "note": "bytes/token/layer measured physical (packed uint8 "
                            "+ fp16 stats; latent coeffs bf16 + residual "
                            "values/indices); fixed latent bases are "
                            "weight-side, amortized, excluded"},
           "rows": []}

for ctx in CONTEXTS:
    for off in OFFSETS:
        ids = window_ids(off, ctx)
        tail = ids[:, -T_EVAL:]
        base, q_list = hy.prefill(ids[:, : ids.shape[1] - T_EVAL + 1],
                                  capture_q=True)
        abar = hy.attention_mass(base, q_list, W)
        S_old = base[0][0].shape[-2] - W
        assert S_old % GROUP == 0, f"S_old {S_old} not divisible by group"

        dense_logits = None
        for name, kc, vc, res, fam in CONFIGS:
            tensors, bpt = compress_asym(base, kc, vc, res, abar)
            cache = hy.build_cache(tensors)
            with torch.no_grad():
                o = model(input_ids=tail, past_key_values=cache, use_cache=True)
            logits = o.logits[0].float()
            del cache, tensors, o
            nll = torch.nn.functional.cross_entropy(
                logits[:-1], tail[0, 1:], reduction="mean").item()
            ppl = float(np.exp(nll))
            if name == "dense":
                dense_logits = logits
            kl = torch.nn.functional.kl_div(
                logits.log_softmax(-1), dense_logits.softmax(-1),
                reduction="none").sum(-1).mean().item()
            results["rows"].append({
                "ctx": ctx, "offset": off, "config": name, "family": fam,
                "bytes_per_tok_layer": bpt, "kl": kl, "ppl": ppl})
            print(f"ctx {ctx:>5} off {off:>6} {name:<20} "
                  f"{bpt:>7.1f} B/tok/layer | KL {kl:8.4f} | PPL {ppl:8.3f}")
            torch.cuda.empty_cache()
        del base, q_list, abar, dense_logits
        import gc
        gc.collect()
        torch.cuda.empty_cache()

with open("results/stage8_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved: results/stage8_results.json")

# ---------------- summary table (mean over offsets) ----------------
print(f"\n{'config':<22} " + "".join(
    f"| ctx {c}: bytes  KL     dPPL% " for c in CONTEXTS))
summary = {}
for name, _, _, _, fam in CONFIGS:
    parts = []
    for ctx in CONTEXTS:
        rr = [r for r in results["rows"]
              if r["config"] == name and r["ctx"] == ctx]
        dd = [r for r in results["rows"]
              if r["config"] == "dense" and r["ctx"] == ctx]
        b = np.mean([r["bytes_per_tok_layer"] for r in rr])
        kl = np.mean([r["kl"] for r in rr])
        dppl = 100 * (np.mean([r["ppl"] for r in rr])
                      / np.mean([r["ppl"] for r in dd]) - 1)
        summary[(name, ctx)] = (b, kl, dppl)
        parts.append(f"| {b:7.1f} {kl:7.4f} {dppl:+6.2f} ")
    print(f"{name:<22} " + "".join(parts))

# ---------------- frontier plot ----------------
FAM = {"quant": ("#eb6834", "s", "packed int2/int4 (KIVI layout)"),
       "oneside": ("#eda100", "v", "one-sided int4"),
       "symlat": ("#1baf7a", "o", "symmetric latent (+res)"),
       "asym": ("#2a78d6", "^", "K-latent + V-int4 (+res)"),
       "vlow": ("#4a3aa7", "D", "aggressive V rank")}
GRAY, INK, MUT = "#9b9a94", "#0b0b0b", "#52514e"
plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
for ax, ctx in zip(axes, CONTEXTS):
    seen_fam = set()
    for name, _, _, _, fam in CONFIGS:
        if fam == "ref":
            continue
        b, kl, dppl = summary[(name, ctx)]
        col, mk, lab = FAM[fam]
        ax.plot(b, max(kl, 1e-3), mk, color=col, ms=8,
                label=lab if fam not in seen_fam else None)
        seen_fam.add(fam)
        short = name.replace("+infK5", "+K res").replace("+inf5", "+res")
        ax.annotate(short, (b, max(kl, 1e-3)), textcoords="offset points",
                    xytext=(6, 3), fontsize=7, color=INK)
    ax.axvline(2048, color=GRAY, ls=":", lw=1.2)
    ax.text(2048, ax.get_ylim()[0], " dense", rotation=90, fontsize=8,
            color=MUT, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured physical bytes / token / layer (K+V, old block)")
    ax.set_ylabel(f"mean KL vs dense over {T_EVAL} positions")
    ax.set_title(f"Packed physical frontier, ctx {ctx} "
                 f"(mean of {len(OFFSETS)} held-out windows)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig("results/stage8_frontier.png", dpi=150)
print("saved: results/stage8_frontier.png")
