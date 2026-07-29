# demo_kv_anatomy.py
# A guided, beginner-friendly LIVE tour of the LLaMA-3.2 KV cache:
#   1. what the model looks like and where K/V come from,
#   2. what the cached K/V tensors are (shapes, memory),
#   3. whether they are LOW RANK (singular values, live compression test),
#   4. whether approximating them changes the model's BEHAVIOR,
#   5. how MATRIX decomposition (per-head SVD) differs from TENSOR
#      decomposition (Tucker / Tensor-Train / CP) on this data.
#
# Run from the repo root:  python experiments/demo_kv_anatomy.py
# (~2-3 minutes on a GPU; prints a narrated walkthrough and saves
#  results/demo_kv_anatomy.png)

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker, tensor_train, parafac
from transformers import DynamicCache
from kv_common import load_model, long_text_ids, svd_compress

tl.set_backend("pytorch")
torch.manual_seed(0)


def banner(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


# ============================================================== 1. the model
banner("STEP 1 -- What does LLaMA-3.2-1B look like?")
tok, model = load_model()
cfg = model.config
L, HID = cfg.num_hidden_layers, cfg.hidden_size
HQ, H, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

print(f"""
The model is a stack of {L} identical TRANSFORMER LAYERS. Each layer has an
attention block and an MLP block. Hidden size: {HID} numbers per token.

Inside attention, three linear maps produce Query, Key, Value from the same
token vector x (shape [{HID}]):""")
lyr = model.model.layers[0].self_attn
print(f"  q_proj weight: {tuple(lyr.q_proj.weight.shape)}  "
      f"-> {HQ} query heads x {D} dims")
print(f"  k_proj weight: {tuple(lyr.k_proj.weight.shape)}  "
      f"-> {H} KEY heads   x {D} dims")
print(f"  v_proj weight: {tuple(lyr.v_proj.weight.shape)}  "
      f"-> {H} VALUE heads x {D} dims")
print(f"""
Note the asymmetry: {HQ} query heads but only {H} K/V heads. This is
GROUPED-QUERY ATTENTION (GQA): each K/V head is SHARED by {HQ // H} query
heads. It already shrinks the cache 4x versus classic multi-head attention.

A ROTARY POSITION EMBEDDING (RoPE) is applied to Q and K (not V): each pair
of dimensions (p, p+{D // 2}) is rotated by an angle that grows with the token's
position. The cache stores keys AFTER this rotation.
""")

# ================================================= 2. catching a real cache
banner("STEP 2 -- Run a real prompt and catch the KV cache")
S = 1024
ids = long_text_ids(tok, S, model.device)     # first 1024 WikiText-2 tokens
cache = DynamicCache(config=cfg)
with torch.no_grad():
    model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)
K0, V0 = cache[0]
Sp = K0.shape[-2]
print(f"""
We fed {Sp} tokens of WikiText-2 through the model once (a "prefill").
During generation the model would re-read these tensors for EVERY new token,
so it caches them. What got cached, per layer:

  K: {tuple(K0.shape)}   V: {tuple(V0.shape)}
     = [batch, kv_heads, sequence_length, head_dim]

Memory arithmetic (bf16 = 2 bytes/number):
  one token, one layer:  K {H}x{D}x2B = 1 KiB, +V = 2 KiB
  x {L} layers          = 32 KiB per token  (all layers)
  this prompt:           {Sp} tokens x 32 KiB = {Sp * 32 / 1024:.0f} MiB
  at the 131072-token context limit: 4 GiB -- MORE than the model's
  own weights (~2.5 GiB). That is the "memory wall" motivating compression.
""")

# ====================================================== 3. is it low rank?
banner("STEP 3 -- Is the cache LOW RANK? (live singular-value test)")
print(f"""
Take one head's keys: a matrix of {Sp} token-rows x {D} dims. "Rank r" means
the matrix can be written as (tall {Sp}x r) @ (wide r x {D}) -- r independent
directions instead of {D}. The SINGULAR VALUES tell us how much 'energy'
each direction carries; if a few dominate, the matrix is compressible.
""")
show_layers = [0, L // 2, L - 1]
spec = {}
print(f"{'layer':>5} | {'K@90%':>6} {'K@95%':>6} {'K@99%':>6} | "
      f"{'V@90%':>6} {'V@95%':>6} {'V@99%':>6}   (rank needed, head-avg)")
for li in range(L):
    K, V = cache[li]
    row = []
    for X in (K, V):
        sv = torch.linalg.svdvals(X[0].float())          # [H, min(S,D)]
        en = (sv**2).cumsum(-1) / (sv**2).sum(-1, keepdim=True)
        row.append([int((en < f).sum(-1).float().mean().item()) + 1
                    for f in (0.90, 0.95, 0.99)])
        if li in show_layers:
            spec.setdefault(li, []).append(sv.mean(0).cpu())  # head-avg spectrum
    if li in (0, 3, 7, 11, 15):
        print(f"{li:>5} | {row[0][0]:>6} {row[0][1]:>6} {row[0][2]:>6} | "
              f"{row[1][0]:>6} {row[1][1]:>6} {row[1][2]:>6}")
print(f"""
READING: Keys need ~20-30 of {D} directions for 95% energy -> moderately low
rank. Values need ~40-50 of {D} -> nearly FULL rank. K and V are different
animals, and any method that compresses them symmetrically ignores this.
""")

# ============================================ 4. does behavior change?
banner("STEP 4 -- Does a rank-r cache change the model's BEHAVIOR?")
print("""
Energy percentages are only a diagnostic. The real question: if we replace
K/V with their best rank-r approximation (per head, all layers) and ask the
model for its next-token probabilities, do they change? We measure
KL(full || compressed) -- 0 means identical; > ~0.5 visibly changes text.
""")
base = [(cache[i][0], cache[i][1]) for i in range(len(cache))]
last = ids[:, -1:]


def next_logits(tensors):
    c = DynamicCache(config=cfg)
    for i, (K, V) in enumerate(tensors):
        c.update(K, V, i)
    with torch.no_grad():
        o = model(input_ids=last, past_key_values=c, use_cache=True)
    return o.logits[0, -1].float()


full = next_logits(base)
behav = []
for r in (8, 16, 32, 48):
    comp = [(svd_compress(K, r).to(K.dtype), svd_compress(V, r).to(V.dtype))
            for K, V in base]
    lr = next_logits(comp)
    kl = torch.nn.functional.kl_div(lr.log_softmax(-1), full.softmax(-1),
                                    reduction="sum").item()
    same = bool((full.argmax() == lr.argmax()).item())
    ratio = (S * D) / (r * (S + D))
    behav.append((r, kl, same, ratio))
    print(f"  rank {r:>2}: KL {kl:8.4f}   top-1 token preserved: {same}   "
          f"(cache {ratio:4.1f}x smaller)")
    del comp
print("""
READING: attractive compression (rank<=16, ~4x) visibly changes behavior;
safe ranks (~32-48) barely compress. Plain low-rank alone is NOT enough --
that is what this repo's stages 4-8 established, and why the working recipe
became: keep recent tokens exact + quantize + repair with a small residual
chosen by ATTENTION INFLUENCE (see RESULTS_AND_NEXT_EXPERIMENTS.md).
""")

# ================================== 5. matrix vs tensor decomposition
banner("STEP 5 -- MATRIX vs TENSOR decomposition (same data, same budget)")
print(f"""
So far we compressed each head's [S x {D}] MATRIX separately -- {H} separate
SVDs per layer. But the layer's cache is really one 3-way TENSOR:

        [ kv_heads x tokens x head_dim ]  =  [{H} x {Sp} x {D}]

A TENSOR decomposition factors all modes jointly, so it can share structure
ACROSS heads (something per-head SVD cannot). Three classic choices:

  Tucker: small core G[r1,r2,r3] + factor per mode      A[{H}xr1] B[{Sp}xr2] C[{D}xr3]
  TT:     chain of 3 'cores', ranks between neighbors   [{H}xR1] [R1x{Sp}xR2] [R2x{D}]
  CP:     sum of R rank-1 outer products                a_r (x) b_r (x) c_r

We give each method roughly the SAME parameter budget and compare
reconstruction error on layer {L // 2}'s keys -- a fair fight, live:
""")
li = L // 2
T = cache[li][0][0].float()                    # [H, S, D]
dense = T.numel()


def relerr(Xh):
    return (torch.norm(T - Xh) / torch.norm(T)).item()


cmp_rows = []
for r in (8, 16):                              # matrix baseline: per-head SVD
    n = H * r * (Sp + D)
    cmp_rows.append(("per-head SVD", f"r={r}", n, relerr(svd_compress(
        T.unsqueeze(0), r)[0])))
for c in ((8, 64, 8), (8, 128, 16)):           # Tucker
    n = c[0] * c[1] * c[2] + H * c[0] + Sp * c[1] + D * c[2]
    core_fac = tucker(T, rank=list(c), init="svd", n_iter_max=20)
    cmp_rows.append(("Tucker", str(c), n, relerr(tl.tucker_to_tensor(core_fac))))
for c in ((8, 8), (8, 16)):                    # Tensor-Train
    n = H * c[0] + c[0] * Sp * c[1] + c[1] * D
    tt = tensor_train(T, rank=[1, c[0], c[1], 1])
    cmp_rows.append(("Tensor-Train", str(c), n, relerr(tl.tt_to_tensor(tt))))
for R in (64, 128):                            # CP
    n = R * (H + Sp + D)
    cp = parafac(T, rank=R, n_iter_max=25, init="random", random_state=0)
    cmp_rows.append(("CP", f"R={R}", n, relerr(tl.cp_to_tensor(cp))))

print(f"{'method':>14} {'config':>14} | {'stored':>8} {'compress':>8} | "
      f"{'rel. error':>10}")
for meth, c, n, e in cmp_rows:
    print(f"{meth:>14} {c:>14} | {n:>8} {dense / n:>7.1f}x | {e:>10.4f}")
print(f"""
READING: at EQUAL budget, Tucker and Tensor-Train do NOT beat per-head SVD
(the mode-by-mode subspaces they add across these {H} heads buy little), but
CP usually EDGES OUT SVD on raw error: its R rank-1 terms are shared across
all heads, so every parameter works on the whole tensor. Two caveats keep
this honest: (i) raw Frobenius error is NOT behavior -- Step 4 showed
behavior is what breaks first; (ii) CP needs an iterative fit (ALS), which
is exactly the online cost a runtime cache codec must pay back. That is why
the SAGE-CP design in references/ fits CP factors OFFLINE over smarter axes
(RoPE pairs: [tokens x heads x {D // 2} pairs x 2]) and adds a sparse residual
for what no smooth factorization captures.
""")

# =================================================================== figure
banner("STEP 6 -- One picture of everything (results/demo_kv_anatomy.png)")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUT = "#0b0b0b", "#52514e"
plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT})
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]                                   # singular-value spectra
for li_, lw in zip(show_layers, (1.2, 2.0, 1.2)):
    svK, svV = spec[li_]
    ax.plot(range(1, D + 1), svK / svK[0], color=BLUE, lw=lw, alpha=0.8,
            label=f"K layer {li_}" if li_ == L // 2 else None)
    ax.plot(range(1, D + 1), svV / svV[0], color=ORANGE, lw=lw, alpha=0.8,
            ls="--", label=f"V layer {li_}" if li_ == L // 2 else None)
ax.set_yscale("log")
ax.set_xlabel("singular value index (of 64)")
ax.set_ylabel("singular value / largest")
ax.set_title("A. Keys decay faster than Values\n(3 layers; head-averaged)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

ax = axes[1]                                   # behavior vs compression
xs = [b[3] for b in behav]
ys = [max(b[1], 1e-3) for b in behav]
ax.plot(xs, ys, "-", color=BLUE, lw=2)
for (r, kl, same, ratio), x, y in zip(behav, xs, ys):
    ax.plot(x, y, "o", ms=8, color=BLUE,
            markerfacecolor=BLUE if same else "white")
    ax.annotate(f"r={r}", (x, y), textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=INK)
ax.axhline(0.5, color=MUT, ls=":", lw=1.2)
ax.text(xs[0], 0.55, "visible behavior change", fontsize=8, color=MUT)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("cache compression (x smaller)")
ax.set_ylabel("KL(full || rank-r) next token")
ax.set_title("B. Useful compression changes behavior\n"
             "(open marker = top-1 token flips)")
ax.grid(alpha=0.25)

ax = axes[2]                                   # matrix vs tensor, equal budget
STYLE = {"per-head SVD": (BLUE, "o"), "Tucker": (ORANGE, "^"),
         "Tensor-Train": (AQUA, "s"), "CP": (YELLOW, "D")}
for meth in STYLE:
    pts = sorted([(dense / n, e) for m, c, n, e in cmp_rows if m == meth])
    col, mk = STYLE[meth]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=1.6,
            marker=mk, ms=8, label=meth)
ax.set_xlabel("compression (x smaller), equal-budget configs")
ax.set_ylabel(f"relative error, layer {L // 2} keys")
ax.set_title("C. Matrix vs tensor decomposition, equal budget\n"
             "(Tucker/TT lose to per-head SVD; CP edges it out)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("results/demo_kv_anatomy.png", dpi=150)
print("saved: results/demo_kv_anatomy.png\n\nDemo complete.")
