# stage6_shufflerless_phase0.py
# Stage 6: "Shufflerless Phase 0 + SAGE residual + Payback", plus first smoke
# tests for CP-Direct/SAGE-CP-KV, SC-KV, FD-KV and a keys-only ablation.
#
# Protocol (extends stages 3-5): Llama-3.2-1B-Instruct on WikiText-2. The last
# T_EVAL=64 tokens of each held-out window are TEACHER-FORCED in one forward
# against (a) the full cache and (b) a compressed cache; we compare per-token
# next-token distributions (mean sum-KL, top-1 agreement rate). The old block
# (all but the last W tokens of the prefill) is compressed; the hot window
# stays exact. Single next-token KL (stages 1-5) was too noisy at this W.
#
#  (A) Shufflerless-CI Phase 0 (proposal ramp step 1): FIXED offline latent
#      bases per (layer, head) calibrated on WikiText-2 TRAIN prefills -- KV
#      "born compressed", no runtime SVD -- vs the per-prompt oracle SVD.
#  (B) SAGE residual: sparse innovations on RoPE-pair groups (p, p+32) picked
#      by the influence surrogate I = abar * qbar * ||e|| (K) / abar * ||e||
#      (V), vs none/random-pair/magnitude-scalar/energy-pair at EQUAL BYTES.
#  (C) Payback admission: measured block coding cost C_fit (fixed GEMM vs
#      runtime SVD vs CP ridge solve) and measured per-token read cost
#      (dense vs latent vs latent+sparse vs CP-direct); admit a block iff
#      N_future * dt_block > 1.1 * C_fit.
#  (D) Representation smoke tests at ctx 2048 (survey ideas): CP bulk with
#      calibrated (B,C,D) factors and online ridge coefficients (SAGE-CP-KV
#      lifecycle step 3), FD-KV streaming Frequent-Directions basis, SC-KV
#      union-of-subspaces (k-means + per-cluster SVD) vs one subspace at equal
#      per-token bytes, and K-only compression (values dense).
#
# RoPE: HF Llama caches keys POST-RoPE; rotate_half pairs dim p with p+D/2.

import gc
import json
import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac
from transformers import DynamicCache
from kv_common import load_model, time_ms

tl.set_backend("pytorch")
torch.manual_seed(0)
tok, model = load_model()
cfg = model.config
L = cfg.num_hidden_layers
HQ = cfg.num_attention_heads
H = cfg.num_key_value_heads
G = HQ // H
D = cfg.head_dim
P = D // 2
DEV = model.device
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
      f"tensorly {tl.__version__}")
print(f"model: {L} layers, {HQ} q-heads, {H} kv-heads, head_dim {D}")

# ---------------- experiment knobs ----------------
W = 64                        # exact hot window
T_EVAL = 64                   # teacher-forced eval tail
CONTEXTS_A = [2048, 8192]
RANKS_A = [8, 16, 32, 48]
N_OFFSETS = 3
CTX_B = 2048
RANKS_B = [16, 32]
RHOS = [0.02, 0.05, 0.10]
CAL_CHUNKS = 8
CAL_CTX = 2048
QBAR_Q = 0.90
BLOCK = 256                   # payback aging-block size
BENCH_S = [2048, 8192, 32768]
N_FUTURE = [64, 256, 1024]
PAIR_BYTES = 2 * 2 + 4
SCAL_BYTES = 2 + 4
CP_RANKS = [128, 256]
CP_CAL_ROWS = 1024            # calibration rows for the CP fit
FD_RANKS = [16, 32]
SC_CFGS = [(4, 8), (4, 16)]   # (clusters, rank per cluster)
KONLY_RANKS = [16, 32]
CALIB_FILE = "results/stage6_calib.pt"

results = {"meta": {
    "model": cfg._name_or_path, "W": W, "t_eval": T_EVAL, "ranks_A": RANKS_A,
    "contexts_A": CONTEXTS_A, "ranks_B": RANKS_B, "rhos": RHOS, "ctx_B": CTX_B,
    "cal_chunks": CAL_CHUNKS, "cal_ctx": CAL_CTX, "qbar_quantile": QBAR_Q,
    "block": BLOCK, "pair_bytes": PAIR_BYTES, "scalar_bytes": SCAL_BYTES,
    "cp_ranks": CP_RANKS, "fd_ranks": FD_RANKS, "sc_cfgs": SC_CFGS,
    "konly_ranks": KONLY_RANKS,
}}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------- data ----------------
from datasets import load_dataset

_SPLITS = {}


def split_ids(split):
    if split not in _SPLITS:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        _SPLITS[split] = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    return _SPLITS[split]


def window_ids(split, offset_tokens, n_tokens):
    """[1, n_tokens] window at offset, BOS prepended (attention sink)."""
    ids = split_ids(split)
    assert ids.shape[1] >= offset_tokens + n_tokens, f"{split} too short"
    if offset_tokens == 0:
        return ids[:, :n_tokens].to(DEV)
    w = ids[:, offset_tokens + 1: offset_tokens + n_tokens]
    bos = torch.full((1, 1), tok.bos_token_id, dtype=ids.dtype)
    return torch.cat([bos, w], dim=1).to(DEV)


# ---------------- query capture (post-RoPE) ----------------
_QCAP = {"on": False, "store": []}


def _hook(module, inp, out):
    if _QCAP["on"]:
        _QCAP["store"].append(out.detach())


for lyr in model.model.layers:
    lyr.self_attn.q_proj.register_forward_hook(_hook)


def rotate_half(x):
    return torch.cat((-x[..., D // 2:], x[..., : D // 2]), dim=-1)


def rope_cos_sin(S):
    pos = torch.arange(S, device=DEV).unsqueeze(0)
    dummy = torch.ones(1, S, 1, dtype=torch.bfloat16, device=DEV)
    cos, sin = model.model.rotary_emb(dummy, pos)
    return cos.float(), sin.float()


def captured_q_postrope(S, rows=None):
    cos, sin = rope_cos_sin(S)
    if rows is not None:
        cos, sin = cos[:, rows], sin[:, rows]
    out = []
    for q in _QCAP["store"]:
        q = q[0] if rows is None else q[0, rows]
        q = q.float().view(-1, HQ, D).transpose(0, 1)
        q = q * cos + rotate_half(q) * sin
        out.append(q)
    return out


def prefill(ids, capture_q=False):
    """Prefill all but the last T_EVAL tokens. Returns (tensors, tail, q)."""
    _QCAP["store"] = []
    _QCAP["on"] = capture_q
    c = DynamicCache(config=cfg)
    with torch.no_grad():
        model(input_ids=ids[:, :-T_EVAL], past_key_values=c, use_cache=True,
              logits_to_keep=1)
    _QCAP["on"] = False
    q_list = captured_q_postrope(ids.shape[1] - T_EVAL) if capture_q else None
    _QCAP["store"] = []
    return [(c[i][0], c[i][1]) for i in range(len(c))], ids[:, -T_EVAL:], q_list


def logits_with(tensors, tail):
    c = DynamicCache(config=cfg)
    for i, (K, V) in enumerate(tensors):
        c.update(K, V, i)
    with torch.no_grad():
        o = model(input_ids=tail, past_key_values=c, use_cache=True)
    out = o.logits[0].float()
    del c, o
    return out                                        # [T_EVAL, vocab]


def kl_top1(full, other):
    """Mean per-position sum-KL and top-1 agreement rate over the eval tail."""
    kl = torch.nn.functional.kl_div(
        other.log_softmax(-1), full.softmax(-1), reduction="none"
    ).sum(-1)
    agree = (full.argmax(-1) == other.argmax(-1)).float()
    return float(kl.mean()), float(agree.mean())


# ---------------- self-test: q pipeline vs eager attention ----------------
print("\n--- self-test: post-RoPE q capture vs eager attention (ctx 128) ---")
ids = window_ids("test", 0, 128)
_QCAP["store"] = []
_QCAP["on"] = True
_impl = model.config._attn_implementation
model.config._attn_implementation = "eager"
with torch.no_grad():
    o = model(input_ids=ids, output_attentions=True, use_cache=True)
model.config._attn_implementation = _impl
_QCAP["on"] = False
S_t = ids.shape[1]
q_all = captured_q_postrope(S_t)
_QCAP["store"] = []
worst = 0.0
for li in (0, L // 2, L - 1):
    K = o.past_key_values[li][0][0].float()
    q = q_all[li][:, -8:]
    sc = torch.einsum("hqd,htd->hqt", q, K.repeat_interleave(G, 0)) / D**0.5
    mask = torch.arange(S_t, device=DEV)[None, :] > \
        (S_t - 8 + torch.arange(8, device=DEV))[:, None]
    probs = sc.masked_fill(mask, -torch.inf).softmax(-1)
    ref = o.attentions[li][0][:, -8:].float()
    worst = max(worst, (probs - ref).abs().max().item())
print(f"max |prob diff| vs eager: {worst:.2e}")
assert worst < 3e-2, "q capture / RoPE application mismatch"
del o, q_all
cleanup()

# ---------------- calibration (offline; cached to disk) ----------------
if os.path.exists(CALIB_FILE):
    print(f"\n--- calibration: loading {CALIB_FILE} ---")
    _c = torch.load(CALIB_FILE, weights_only=True)
    UFULL, QBAR, XCAL = _c["UFULL"].to(DEV), _c["QBAR"].to(DEV), _c["XCAL"]
else:
    print(f"\n--- calibration: {CAL_CHUNKS} x {CAL_CTX} train tokens ---")
    M = torch.zeros(L, 2, H, D, D, device=DEV)
    q_pair_samples = [[] for _ in range(L)]
    xc = [[[], []] for _ in range(L)]
    rows_per_chunk = CP_CAL_ROWS // CAL_CHUNKS
    for c_i in range(CAL_CHUNKS):
        ids = window_ids("train", c_i * 4096, CAL_CTX)
        base, _, q_list = prefill(ids, capture_q=True)
        S_c = base[0][0].shape[-2]
        qrows = torch.arange(0, S_c, 8, device=DEV)
        crows = torch.linspace(0, S_c - 1, rows_per_chunk).long()
        for li, (K, V) in enumerate(base):
            Kf, Vf = K[0].float(), V[0].float()
            M[li, 0] += torch.einsum("hsd,hse->hde", Kf, Kf)
            M[li, 1] += torch.einsum("hsd,hse->hde", Vf, Vf)
            q = q_list[li][:, qrows]
            pn = (q[..., :P] ** 2 + q[..., P:] ** 2).sqrt()
            q_pair_samples[li].append(
                pn.view(H, G, -1, P).permute(0, 2, 1, 3).reshape(H, -1, P).cpu())
            xc[li][0].append(Kf[:, crows].to(torch.float16).cpu())
            xc[li][1].append(Vf[:, crows].to(torch.float16).cpu())
        del base, q_list
        cleanup()
    eigval, eigvec = torch.linalg.eigh(M)
    UFULL = eigvec.flip(-1).contiguous()               # [L, 2, H, D, D]
    QBAR = torch.stack([
        torch.quantile(torch.cat(s, dim=1).float(), QBAR_Q, dim=1)
        for s in q_pair_samples]).to(DEV)              # [L, H, P]
    # CP calibration rows: [L, 2, H, n_cal, D] fp16 on CPU
    XCAL = torch.stack([torch.stack([torch.cat(t, dim=1) for t in lt])
                        for lt in xc])
    torch.save({"UFULL": UFULL.cpu(), "QBAR": QBAR.cpu(), "XCAL": XCAL},
               CALIB_FILE)
    del M, q_pair_samples, xc, eigvec, eigval
    cleanup()
print(f"calibration ready: UFULL {tuple(UFULL.shape)}, QBAR {tuple(QBAR.shape)}, "
      f"XCAL {tuple(XCAL.shape)}")


def project_fixed(X, li, ti, r):
    U = UFULL[li, ti, :, :, :r]
    A = torch.einsum("hsd,hdr->hsr", X, U)
    return torch.einsum("hsr,hdr->hsd", A, U), A


def project_oracle(X, r):
    U, sv, Vh = torch.linalg.svd(X, full_matrices=False)
    return (U[..., :r] * sv[..., :r].unsqueeze(-2)) @ Vh[..., :r, :]


def compress_cache(base, r, variant, residual=None, parts=("K", "V"),
                   custom=None):
    """Rebuild (K, V) with the old block compressed. variant: fixed|oracle.
    custom: optional fn(li, Kf, Vf) -> (Khat, Vhat) overriding the projector."""
    out = []
    for li, (K, V) in enumerate(base):
        S = K.shape[-2]
        Kf, Vf = K[0, :, : S - W].float(), V[0, :, : S - W].float()
        if custom is not None:
            Kh, Vh = custom(li, Kf, Vf)
        else:
            if "K" in parts:
                Kh = project_fixed(Kf, li, 0, r)[0] if variant == "fixed" \
                    else project_oracle(Kf, r)
            else:
                Kh = Kf
            if "V" in parts:
                Vh = project_fixed(Vf, li, 1, r)[0] if variant == "fixed" \
                    else project_oracle(Vf, r)
            else:
                Vh = Vf
        if residual is not None:
            Kh, Vh = Kh + residual[li][0], Vh + residual[li][1]
        Kc = torch.cat([Kh.unsqueeze(0).to(K.dtype), K[:, :, S - W:]], dim=-2)
        Vc = torch.cat([Vh.unsqueeze(0).to(V.dtype), V[:, :, S - W:]], dim=-2)
        out.append((Kc, Vc))
    return out


def rel_err(base, comp):
    ek, ev = [], []
    for (K, V), (Kc, Vc) in zip(base, comp):
        ek.append((torch.norm(K.float() - Kc.float()) / torch.norm(K.float())).item())
        ev.append((torch.norm(V.float() - Vc.float()) / torch.norm(V.float())).item())
    return float(np.mean(ek)), float(np.mean(ev))


def evaluate(base, tail, full, **kw):
    comp = compress_cache(base, kw.pop("r", 0), kw.pop("variant", "fixed"), **kw)
    kl, ag = kl_top1(full, logits_with(comp, tail))
    ek, ev = rel_err(base, comp)
    del comp
    cleanup()
    return kl, ag, ek, ev


# ---------------- Part A ----------------
print("\n=== Part A: Shufflerless Phase 0 -- fixed basis vs per-prompt oracle ===")
results["phase0"] = []
TEST_OFF = [0, 40000, 80000]

for ctx in CONTEXTS_A:
    for off in TEST_OFF[:N_OFFSETS]:
        ids = window_ids("test", off, ctx)
        base, tail, _ = prefill(ids)
        full = logits_with(base, tail)
        S_old = base[0][0].shape[-2] - W

        if ctx == CONTEXTS_A[0] and off == 0:
            kl, ag, _, _ = evaluate(base, tail, full, r=D, variant="fixed")
            print(f"sanity r={D} fixed: KL {kl:.2e} top1 {ag:.3f}")
            # KL is the losslessness gate; top-1 can flip on near-ties in bf16
            assert kl < 2e-2 and ag > 0.9, "full-rank fixed basis not lossless"

        for r in RANKS_A:
            row = {"ctx": ctx, "offset": off, "rank": r, "S_old": S_old}
            for variant in ("fixed", "oracle"):
                kl, ag, ek, ev = evaluate(base, tail, full, r=r, variant=variant)
                row[variant] = {"kl": kl, "top1": ag, "rel_err_K": ek,
                                "rel_err_V": ev}
            row["compression_old"] = D / r
            results["phase0"].append(row)
            print(f"ctx {ctx:>5} off {off:>6} r={r:<3}"
                  f" | fixed KL {row['fixed']['kl']:8.4f} top1 {row['fixed']['top1']:.3f}"
                  f" relK {row['fixed']['rel_err_K']:.3f}"
                  f" | oracle KL {row['oracle']['kl']:8.4f} top1 {row['oracle']['top1']:.3f}"
                  f" relK {row['oracle']['rel_err_K']:.3f}")
        del base, full, tail
        cleanup()
        print(f"  [mem] allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

# ---------------- Part B: SAGE residual ----------------
print("\n=== Part B: SAGE influence residual vs baselines (equal bytes) ===")
results["sage"] = []
results["concentration"] = []
RULES = ["random", "magnitude", "energy", "influence"]


def attention_mass(base, q_list, S_p):
    out = []
    qpos = S_p - W + torch.arange(W, device=DEV)
    mask = torch.arange(S_p, device=DEV)[None, :] > qpos[:, None]
    for li, (K, V) in enumerate(base):
        q = q_list[li][:, -W:]
        sc = torch.einsum("hqd,htd->hqt", q,
                          K[0].float().repeat_interleave(G, 0)) / D**0.5
        pr = sc.masked_fill(mask[None], -torch.inf).softmax(-1)
        out.append(pr.view(H, G, W, S_p).mean(dim=(1, 2))[:, : S_p - W])
    return out


def pick_pairs(score, m):
    flat = score.reshape(-1)
    idx = torch.topk(flat, min(m, flat.numel())).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view(score.shape)


for r in RANKS_B:
    for off in TEST_OFF[:N_OFFSETS]:
        ids = window_ids("test", off, CTX_B)
        base, tail, q_list = prefill(ids, capture_q=True)
        full = logits_with(base, tail)
        S_p = base[0][0].shape[-2]
        S_old = S_p - W
        abar = attention_mass(base, q_list, S_p)

        E_K, E_V, pnK, pnV = [], [], [], []
        for li, (K, V) in enumerate(base):
            Kf, Vf = K[0, :, :S_old].float(), V[0, :, :S_old].float()
            ek = Kf - project_fixed(Kf, li, 0, r)[0]
            ev = Vf - project_fixed(Vf, li, 1, r)[0]
            E_K.append(ek)
            E_V.append(ev)
            pnK.append((ek[..., :P] ** 2 + ek[..., P:] ** 2 + 1e-20).sqrt())
            pnV.append((ev[..., :P] ** 2 + ev[..., P:] ** 2 + 1e-20).sqrt())

        kl0, ag0, _, _ = evaluate(base, tail, full, r=r, variant="fixed")
        results["sage"].append({"rank": r, "offset": off, "rule": "none",
                                "rho": 0.0, "bytes_per_layer": 0, "kl": kl0,
                                "top1": ag0})
        print(f"r={r} off {off}: latent-only KL {kl0:.4f} top1 {ag0:.3f}")

        if off == 0:
            for li in range(L):
                en = (pnK[li] ** 2).reshape(-1).sort(descending=True).values
                cum = (en.cumsum(0) / en.sum()).cpu()
                fr = [float(cum[max(1, int(f * cum.numel())) - 1])
                      for f in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)]
                results["concentration"].append({"rank": r, "layer": li,
                                                 "energy_at_frac": fr})

        n_groups = H * S_old * P
        for rho in RHOS:
            m_pairs = int(rho * n_groups)
            budget = m_pairs * PAIR_BYTES
            m_scal = budget // SCAL_BYTES
            for rule in RULES:
                res = []
                gen = torch.Generator().manual_seed(off + 17)
                for li in range(L):
                    if rule == "magnitude":
                        pair = []
                        for E in (E_K[li], E_V[li]):
                            flat = E.reshape(-1)
                            keep = torch.zeros_like(flat)
                            isel = torch.topk(flat.abs(), m_scal).indices
                            keep[isel] = flat[isel]
                            pair.append(keep.view(E.shape))
                        res.append((pair[0], pair[1]))
                        continue
                    if rule == "random":
                        perm = torch.randperm(n_groups, generator=gen)[:m_pairs]
                        mk = torch.zeros(n_groups, dtype=torch.bool)
                        mk[perm] = True
                        mk = mk.view(H, S_old, P).to(DEV)
                        mv = mk
                    elif rule == "energy":
                        mk = pick_pairs(pnK[li], m_pairs)
                        mv = pick_pairs(pnV[li], m_pairs)
                    else:  # influence (SAGE)
                        a = abar[li].unsqueeze(-1)
                        mk = pick_pairs(a * QBAR[li].unsqueeze(1) * pnK[li], m_pairs)
                        mv = pick_pairs(a * pnV[li], m_pairs)
                    keepK = torch.cat([mk, mk], dim=-1)
                    keepV = torch.cat([mv, mv], dim=-1)
                    res.append((E_K[li] * keepK, E_V[li] * keepV))
                kl, ag, _, _ = evaluate(base, tail, full, r=r, variant="fixed",
                                        residual=res)
                del res
                results["sage"].append({
                    "rank": r, "offset": off, "rule": rule, "rho": rho,
                    "bytes_per_layer": int(budget * 2), "kl": kl, "top1": ag})
                print(f"r={r} off {off} rho {rho:.2f} {rule:<10} "
                      f"KL {kl:8.4f} top1 {ag:.3f}")
        del base, full, tail, q_list, E_K, E_V, pnK, pnV, abar
        cleanup()

# ---------------- Part D: representation smoke tests (ctx 2048) ----------------
print("\n=== Part D: smoke tests -- CP / SC-KV / FD-KV / K-only ===")
results["smoke"] = []


def fd_basis(X, r, chunk=256):
    """Streaming Frequent-Directions sketch -> basis [H, D, r]."""
    ell = 2 * r
    sk = None
    for c0 in range(0, X.shape[1], chunk):
        blk = X[:, c0:c0 + chunk]
        sk = blk if sk is None else torch.cat([sk, blk], dim=1)
        if sk.shape[1] >= ell:
            _, sv, Vh = torch.linalg.svd(sk, full_matrices=False)
            keep = min(ell - 1, sv.shape[-1] - 1)
            delta = (sv[:, keep] ** 2).unsqueeze(-1)
            s2 = (sv[:, :keep] ** 2 - delta).clamp_min(0).sqrt()
            sk = s2.unsqueeze(-1) * Vh[:, :keep, :]
    _, sv, Vh = torch.linalg.svd(sk, full_matrices=False)
    return Vh[:, :r, :].transpose(1, 2).contiguous()


def fd_project(X, r):
    U = fd_basis(X, r)
    return torch.einsum("hsr,hdr->hsd", torch.einsum("hsd,hdr->hsr", X, U), U)


def sc_project(Kf, Vf, k, r, seed):
    """k-means on keys (shared token clusters), per-cluster SVD for K and V."""
    Kh, Vh = torch.empty_like(Kf), torch.empty_like(Vf)
    gen = torch.Generator().manual_seed(seed)
    S_ = Kf.shape[1]
    for h in range(H):
        x = Kf[h]
        cent = x[torch.randperm(S_, generator=gen)[:k].to(x.device)].clone()
        for _ in range(10):
            asg = torch.cdist(x, cent).argmin(-1)
            for j in range(k):
                sel = asg == j
                if sel.any():
                    cent[j] = x[sel].mean(0)
        for j in range(k):
            sel = asg == j
            if not sel.any():
                continue
            for X, out in ((Kf[h], Kh[h]), (Vf[h], Vh[h])):
                U, sv, Vt = torch.linalg.svd(X[sel], full_matrices=False)
                rr = min(r, sv.shape[0])
                out[sel] = (U[:, :rr] * sv[:rr]) @ Vt[:rr]
    return Kh, Vh


# --- CP fit on calibration rows (SAGE-CP lifecycle steps 1+3) ---
print("fitting CP factors on calibration rows "
      f"(n={XCAL.shape[3]}, ranks {CP_RANKS}) ...")
CPFAC = {}    # (R, ti) -> per-layer list of (M [H*P*2, R], Mridge [H*P*2, R])
for R in CP_RANKS:
    for ti in range(2):
        per_layer = []
        for li in range(L):
            X = XCAL[li, ti].to(DEV).float()               # [H, n, D]
            T = torch.stack([X[..., :P], X[..., P:]], dim=-1).permute(1, 0, 2, 3)
            _, fac = parafac(T.contiguous(), rank=R, n_iter_max=15,
                             init="random", random_state=0, tol=1e-7)
            Mkr = tl.tenalg.khatri_rao(fac[1:])            # [(H P 2), R]
            Gm = Mkr.T @ Mkr
            Gm += 1e-4 * Gm.diagonal().mean() * torch.eye(R, device=DEV)
            per_layer.append((Mkr, Mkr @ torch.linalg.inv(Gm)))
        CPFAC[(R, ti)] = per_layer
        print(f"  CP R={R} tensor={'KV'[ti]} fitted ({L} layers)")
        cleanup()


def cp_project(X, li, ti, R):
    """X [H, S, D] -> CP reconstruction via ridge token coefficients."""
    Mkr, Mridge = CPFAC[(R, ti)][li]
    T = torch.stack([X[..., :P], X[..., P:]], dim=-1).permute(1, 0, 2, 3)
    unf = T.reshape(T.shape[0], -1)                        # [S, H*P*2]
    A = unf @ Mridge
    rec = (A @ Mkr.T).view(T.shape)                        # [S, H, P, 2]
    return torch.cat([rec[..., 0], rec[..., 1]], dim=-1).permute(1, 0, 2), A


SMOKE = (
    [("fixed-K", {"r": r, "variant": "fixed", "parts": ("K",)},
      (H * r + H * D) * 2) for r in KONLY_RANKS]
    + [("oracle-K", {"r": r, "variant": "oracle", "parts": ("K",)},
       (H * r + H * D) * 2) for r in KONLY_RANKS]
    + [("fd", {"custom_name": "fd", "r": r}, 2 * H * r * 2) for r in FD_RANKS]
    + [("sc", {"custom_name": "sc", "k": k, "r": r}, 2 * H * r * 2 + 4)
       for (k, r) in SC_CFGS]
    + [("cp", {"custom_name": "cp", "r": R}, 2 * R * 2) for R in CP_RANKS]
)

for off in TEST_OFF[:N_OFFSETS]:
    ids = window_ids("test", off, CTX_B)
    base, tail, _ = prefill(ids)
    full = logits_with(base, tail)
    for name, kw, bpt in SMOKE:
        kw = dict(kw)
        cname = kw.pop("custom_name", None)
        if cname == "fd":
            r = kw.pop("r")
            custom = lambda li, Kf, Vf: (fd_project(Kf, r), fd_project(Vf, r))
        elif cname == "sc":
            k, r = kw.pop("k"), kw.pop("r")
            custom = lambda li, Kf, Vf: sc_project(Kf, Vf, k, r, seed=li + off)
        elif cname == "cp":
            r = kw.pop("r")
            custom = lambda li, Kf, Vf: (cp_project(Kf, li, 0, r)[0],
                                         cp_project(Vf, li, 1, r)[0])
        else:
            custom = None
        kl, ag, ek, ev = evaluate(base, tail, full, custom=custom, **kw)
        cfg_r = kw.get("r", r)
        results["smoke"].append({"method": name, "rank": cfg_r, "offset": off,
                                 "coeff_bytes_per_tok_layer": bpt, "kl": kl,
                                 "top1": ag, "rel_err_K": ek, "rel_err_V": ev})
        print(f"off {off} {name:<9} r={cfg_r:<4} bytes/tok/layer {bpt:>5} | "
              f"KL {kl:8.4f} top1 {ag:.3f} relK {ek:.3f} relV {ev:.3f}")
    del base, full, tail
    cleanup()

# ---------------- Part C: payback ----------------
print("\n=== Part C: payback -- coding cost vs per-token read savings ===")
results["payback"] = {}
R0 = 16
R0_CP = 128
RHO0 = 0.05

ids = window_ids("test", 0, CTX_B)
base, tail, _ = prefill(ids)
XK = torch.stack([b[0][0, :, :BLOCK].float() for b in base])
XV = torch.stack([b[1][0, :, :BLOCK].float() for b in base])
del base, tail
cleanup()
m_blk = int(RHO0 * H * BLOCK * P)


def residual_select(X, Xh):
    E = X - Xh
    pn = (E[..., :P] ** 2 + E[..., P:] ** 2).sqrt()
    idx = torch.topk(pn.reshape(L, -1), m_blk, dim=-1).indices
    torch.gather(E[..., :P].reshape(L, -1), 1, idx)
    torch.gather(E[..., P:].reshape(L, -1), 1, idx)


def code_fixed():
    U_K, U_V = UFULL[:, 0, :, :, :R0], UFULL[:, 1, :, :, :R0]
    AK = torch.einsum("lhsd,lhdr->lhsr", XK, U_K)
    AV = torch.einsum("lhsd,lhdr->lhsr", XV, U_V)
    residual_select(XK, torch.einsum("lhsr,lhdr->lhsd", AK, U_K))
    residual_select(XV, torch.einsum("lhsr,lhdr->lhsd", AV, U_V))


def code_svd():
    for X in (XK, XV):
        U, sv, Vh = torch.linalg.svd(X.reshape(L * H, BLOCK, D),
                                     full_matrices=False)
        Xh = (U[..., :R0] * sv[..., :R0].unsqueeze(-2)) @ Vh[..., :R0, :]
        residual_select(X, Xh.reshape(L, H, BLOCK, D))


MR_K = torch.stack([CPFAC[(R0_CP, 0)][li][1] for li in range(L)])
MR_V = torch.stack([CPFAC[(R0_CP, 1)][li][1] for li in range(L)])


def code_cp():
    for X, MR in ((XK, MR_K), (XV, MR_V)):
        T = torch.stack([X[..., :P], X[..., P:]], dim=-1)   # [L,H,B,P,2]
        unf = T.permute(0, 2, 1, 3, 4).reshape(L, BLOCK, -1)
        torch.bmm(unf, MR)


t_fixed = time_ms(code_fixed, warmup=3, iters=10)
t_svd = time_ms(code_svd, warmup=3, iters=10)
t_cp = time_ms(code_cp, warmup=3, iters=10)
print(f"C_fit per {BLOCK}-token block (all layers, K+V): fixed(r={R0}) "
      f"{t_fixed['median']:.2f} ms | runtime-SVD {t_svd['median']:.2f} ms | "
      f"CP ridge (R={R0_CP}) {t_cp['median']:.2f} ms")
results["payback"]["c_fit_ms"] = {"fixed": t_fixed, "svd": t_svd, "cp": t_cp,
                                  "block": BLOCK, "rank": R0, "rho": RHO0,
                                  "cp_rank": R0_CP}
del XK, XV
cleanup()

bench = []
for S in BENCH_S:
    Kd = torch.randn(L * H, S, D, device=DEV, dtype=torch.bfloat16)
    Vd = torch.randn_like(Kd)
    AK = torch.randn(L * H, S, R0, device=DEV, dtype=torch.bfloat16)
    AV = torch.randn_like(AK)
    U_K = torch.randn(L * H, D, R0, device=DEV, dtype=torch.bfloat16)
    U_V = torch.randn_like(U_K)
    qg = torch.randn(L * H, G, D, device=DEV, dtype=torch.bfloat16)
    # CP: token factor shared across heads
    Acp = torch.randn(L, S, R0_CP, device=DEV, dtype=torch.bfloat16)
    Acpv = torch.randn_like(Acp)
    Bf = torch.randn(L, H, R0_CP, device=DEV, dtype=torch.bfloat16)
    Cf = torch.randn(L, P, R0_CP, device=DEV, dtype=torch.bfloat16)
    Df = torch.randn(L, 2, R0_CP, device=DEV, dtype=torch.bfloat16)
    qp = torch.randn(L, H, P, 2, device=DEV, dtype=torch.bfloat16)

    def step_dense():
        sc = torch.bmm(qg, Kd.transpose(1, 2)) / D**0.5
        pr = sc.float().softmax(-1).to(torch.bfloat16)
        torch.bmm(pr, Vd)

    def step_latent():
        sc = torch.bmm(torch.bmm(qg, U_K), AK.transpose(1, 2)) / D**0.5
        pr = sc.float().softmax(-1).to(torch.bfloat16)
        torch.bmm(torch.bmm(pr, AV), U_V.transpose(1, 2))

    def step_cp():
        zq = torch.einsum("lhpu,lpr,lur->lhr", qp, Cf, Df) * Bf
        sc = torch.bmm(Acp, zq.transpose(1, 2)) / D**0.5   # [L, S, H]
        pr = sc.float().softmax(1).to(torch.bfloat16)
        g = torch.einsum("lsh,lsr->lhr", pr, Acpv)
        torch.einsum("lhr,lpr,lur->lhpu", g * Bf, Cf, Df)

    csr, csr_t = [], []
    nnz_pairs = int(RHO0 * H * S * P)
    for li in range(L):
        gsel = torch.randperm(H * S * P, device=DEV)[:nnz_pairs]
        h = gsel // (S * P)
        s = (gsel // P) % S
        p = gsel % P
        rows = torch.cat([h * S + s, h * S + s])
        cols = torch.cat([h * D + p, h * D + p + P])
        vals = torch.randn(2 * nnz_pairs, device=DEV)
        E = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals,
                                    (H * S, H * D)).coalesce()
        csr.append(E.to_sparse_csr())
        csr_t.append(E.t().coalesce().to_sparse_csr())
    qs = torch.randn(L, H * D, G, device=DEV)
    als = torch.randn(L, H * S, G, device=DEV)

    def step_sparse_corr():
        for li in range(L):
            torch.sparse.mm(csr[li], qs[li])
            torch.sparse.mm(csr_t[li], als[li])

    td = time_ms(step_dense, warmup=5, iters=20)["median"]
    tl_ = time_ms(step_latent, warmup=5, iters=20)["median"]
    tc = time_ms(step_cp, warmup=5, iters=20)["median"]
    ts = time_ms(step_sparse_corr, warmup=5, iters=20)["median"]
    bench.append({"S": S, "dense_ms": td, "latent_ms": tl_,
                  "cp_ms": tc, "sparse_corr_ms": ts})
    print(f"S={S:>6}: dense {td:7.3f} | latent r{R0} {tl_:7.3f} | "
          f"CP R{R0_CP} {tc:7.3f} | sparse corr {ts:7.3f} ms (unfused CSR)")
    del Kd, Vd, AK, AV, U_K, U_V, qg, csr, csr_t, qs, als
    del Acp, Acpv, Bf, Cf, Df, qp
    cleanup()
results["payback"]["decode_step"] = bench

CODINGS = {"fixed": ("latent", "latent_ms"),
           "svd": ("latent", "latent_ms"),
           "cp": ("cp-direct", "cp_ms")}
decisions = []
for b in bench:
    frac = BLOCK / b["S"]
    for coding, cfit in (("fixed", t_fixed["median"]), ("svd", t_svd["median"]),
                         ("cp", t_cp["median"])):
        variants = [(CODINGS[coding][0], b[CODINGS[coding][1]])]
        if coding in ("fixed", "svd"):
            variants.append(("latent+sparse",
                             b["latent_ms"] + b["sparse_corr_ms"]))
        for vname, t_c in variants:
            dt_blk = (b["dense_ms"] - t_c) * frac
            n_be = float("inf") if dt_blk <= 0 else 1.1 * cfit / dt_blk
            for nf in N_FUTURE:
                decisions.append({
                    "S": b["S"], "coding": coding, "variant": vname,
                    "n_future": nf, "dt_block_ms": dt_blk,
                    "n_breakeven": None if n_be == float("inf") else n_be,
                    "admit": bool(nf > n_be)})
results["payback"]["decisions"] = decisions
print("\npayback admission (admit iff N_future * dt_block > 1.1 * C_fit):")
print(f"{'S':>6} {'coding':>6} {'variant':>14} {'dt_blk ms':>10} "
      f"{'N_break':>9} | " + " ".join(f"N={n}" for n in N_FUTURE))
seen = set()
for d in decisions:
    key = (d["S"], d["coding"], d["variant"])
    if key in seen:
        continue
    seen.add(key)
    row = [x for x in decisions if (x["S"], x["coding"], x["variant"]) == key]
    nb = row[0]["n_breakeven"]
    admits = " ".join("Y" if x["admit"] else "n" for x in row)
    print(f"{d['S']:>6} {d['coding']:>6} {d['variant']:>14} "
          f"{row[0]['dt_block_ms']:>10.4f} "
          f"{'inf' if nb is None else format(nb, '9.1f')} | {admits}")

with open("results/stage6_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved: results/stage6_results.json")

# ---------------- plots ----------------
BLUE, ORANGE, AQUA, YELLOW, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a",
                                      "#eda100", "#4a3aa7")
SEQ = ["#86b6ef", "#2a78d6", "#0d366b"]      # sequential blue steps
GRAY, INK, MUT = "#9b9a94", "#0b0b0b", "#52514e"
plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT})
fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.6))


def mean_over_offsets(rows, key_kl, key_t1):
    kl = max(float(np.mean([r[key_kl] for r in rows])), 1e-3)
    t1 = float(np.mean([r[key_t1] for r in rows]))
    return kl, t1


# P1: Part A
ax = axes[0, 0]
for ctx, ls in ((2048, "--"), (8192, "-")):
    for variant, col in (("fixed", BLUE), ("oracle", ORANGE)):
        rows = [p for p in results["phase0"] if p["ctx"] == ctx]
        xs = sorted({p["rank"] for p in rows})
        ys, fl = [], []
        for rk in xs:
            rr = [p[variant] for p in rows if p["rank"] == rk]
            k, t = mean_over_offsets(rr, "kl", "top1")
            ys.append(k)
            fl.append(t >= 0.95)
        ax.plot(xs, ys, ls, color=col, lw=2, label=f"{variant}, ctx {ctx}")
        for x, y, f_ in zip(xs, ys, fl):
            ax.plot(x, y, "o", color=col, ms=7,
                    markerfacecolor=col if f_ else "white")
ax.set_yscale("log")
ax.set_xlabel("rank r (old block; hot window exact)")
ax.set_ylabel("mean KL over 64 held-out positions")
ax.set_title("A. Fixed offline basis vs per-prompt oracle SVD\n"
             "(filled = top-1 agreement >= 95%)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

# P2: Part B
ax = axes[0, 1]
RULE_STYLE = {"influence": (BLUE, "SAGE influence pair"),
              "energy": (ORANGE, "energy pair"),
              "magnitude": (AQUA, "magnitude scalar"),
              "random": (YELLOW, "random pair")}
for r, ls in ((16, "-"), (32, "--")):
    for rule in ("influence", "energy", "magnitude", "random"):
        col, lab = RULE_STYLE[rule]
        xs, ys = [], []
        for rho in RHOS:
            rr = [s for s in results["sage"]
                  if s["rank"] == r and s["rule"] == rule and s["rho"] == rho]
            xs.append(100 * rho)
            ys.append(max(np.mean([x["kl"] for x in rr]), 1e-3))
        ax.plot(xs, ys, ls, color=col, lw=2, marker="o", ms=6,
                label=lab if r == RANKS_B[0] else None)
    b_kl = np.mean([s["kl"] for s in results["sage"]
                    if s["rank"] == r and s["rule"] == "none"])
    ax.axhline(max(b_kl, 1e-3), color=GRAY, ls=ls, lw=1.4)
    ax.text(RHOS[0] * 100, b_kl * 1.1, f"latent-only r={r}", fontsize=8,
            color=MUT)
ax.set_yscale("log")
ax.set_xlabel("residual budget, % of old-block RoPE-pair groups (equal bytes)")
ax.set_ylabel("mean KL over 64 positions")
ax.set_title("B. SAGE residual vs baselines at equal bytes\n"
             "(solid r=16, dashed r=32)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

# P3: smoke-test Pareto
ax = axes[0, 2]
METH = {  # method -> (color, marker, label, open?)
    "fixed-KV": (BLUE, "o", "fixed basis KV", False),
    "oracle-KV": (ORANGE, "^", "oracle SVD KV", False),
    "fixed-K": (BLUE, "o", "fixed K-only", True),
    "oracle-K": (ORANGE, "^", "oracle K-only", True),
    "fd": (AQUA, "s", "FD-KV streaming", False),
    "sc": (VIOLET, "v", "SC-KV k=4", False),
    "cp": (YELLOW, "D", "CP (SAGE-CP bulk)", False),
}
smoke_pts = {}
for m in METH:
    if m in ("fixed-KV", "oracle-KV"):
        variant = m.split("-")[0]
        for r in RANKS_A:
            rows = [p[variant] for p in results["phase0"]
                    if p["ctx"] == CTX_B and p["rank"] == r]
            k, _ = mean_over_offsets(rows, "kl", "top1")
            smoke_pts.setdefault(m, []).append((2 * H * r * 2, k))
    else:
        rows_m = [s for s in results["smoke"] if s["method"] == m]
        for r in sorted({s["rank"] for s in rows_m}):
            rr = [s for s in rows_m if s["rank"] == r]
            k, _ = mean_over_offsets(rr, "kl", "top1")
            smoke_pts.setdefault(m, []).append(
                (rr[0]["coeff_bytes_per_tok_layer"], k))
for m, pts in smoke_pts.items():
    col, mk, lab, opn = METH[m]
    pts = sorted(pts)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=1.4,
            alpha=0.7, label=lab)
    for x, y in pts:
        ax.plot(x, y, mk, color=col, ms=7,
                markerfacecolor="white" if opn else col)
    ax.annotate(lab, pts[-1], textcoords="offset points", xytext=(5, 4),
                fontsize=7.5, color=INK)
ax.legend(fontsize=7)
ax.axvline(2 * H * D * 2, color=GRAY, ls=":", lw=1.2)
ax.text(2 * H * D * 2, ax.get_ylim()[0] * 1.5 if ax.get_ylim()[0] > 0 else 1e-3,
        " dense", fontsize=8, color=MUT, rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("coefficient bytes / token / layer (K+V)")
ax.set_ylabel("mean KL over 64 positions")
ax.set_title(f"C. Representation smoke tests, ctx {CTX_B}\n"
             "(open marker = keys-only, values dense)")
ax.grid(alpha=0.25)

# P4: C_fit bars
ax = axes[1, 0]
bars = [(f"fixed basis\nr={R0}", t_fixed["median"], BLUE),
        (f"runtime SVD\nr={R0}", t_svd["median"], ORANGE),
        (f"CP ridge\nR={R0_CP}", t_cp["median"], YELLOW)]
for i, (lab, v, col) in enumerate(bars):
    ax.bar(i, v, 0.55, color=col)
    ax.text(i, v * 1.06, f"{v:.2f} ms", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(len(bars)))
ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.set_yscale("log")
ax.set_ylabel(f"C_fit per {BLOCK}-token block, all layers K+V (ms)")
ax.set_title("D. Block coding cost (what Phase 0 removes)")
ax.grid(alpha=0.25, axis="y")

# P5: breakeven
ax = axes[1, 1]
for coding, col in (("fixed", BLUE), ("svd", ORANGE), ("cp", YELLOW)):
    for vname, ls in (("latent", "-"), ("latent+sparse", ":"),
                      ("cp-direct", "-")):
        pts = [(b["S"], next((d["n_breakeven"] for d in decisions
                              if d["S"] == b["S"] and d["coding"] == coding
                              and d["variant"] == vname), None))
               for b in bench]
        pts = [(s, n) for s, n in pts
               if any(d["S"] == s and d["coding"] == coding
                      and d["variant"] == vname for d in decisions)]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] if p[1] is not None else np.nan for p in pts]
        ax.plot(xs, ys, ls, color=col, lw=2, marker="o", ms=6,
                label=f"{coding} coding, {vname}")
for nf in N_FUTURE:
    ax.axhline(nf, color=GRAY, ls="--", lw=1)
    ax.text(BENCH_S[0], nf * 1.12, f"N_future={nf}", fontsize=8, color=MUT)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("context length S at coding time")
ax.set_ylabel("breakeven future tokens (admit above line)")
ax.set_title("E. Payback: decode tokens needed to repay coding")
ax.grid(alpha=0.25)
ax.legend(fontsize=7.5)

# P6: residual concentration (sequential ramp, one hue)
ax = axes[1, 2]
fracs = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
for f_i, col in zip((0, 2, 3), SEQ):
    rows = [c for c in results["concentration"] if c["rank"] == RANKS_B[0]]
    xs = [c["layer"] for c in rows]
    ys = [c["energy_at_frac"][f_i] for c in rows]
    ax.plot(xs, ys, "-", color=col, lw=2, marker="o", ms=5,
            label=f"top {fracs[f_i] * 100:.0f}% of pair groups")
ax.set_xlabel("layer")
ax.set_ylabel("fraction of K-residual energy captured")
ax.set_title(f"F. Residual concentration per layer (r={RANKS_B[0]})\n"
             "how 'spiky' is what the fixed basis misses?")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("results/stage6_phase0.png", dpi=150)
print("saved: results/stage6_phase0.png")
