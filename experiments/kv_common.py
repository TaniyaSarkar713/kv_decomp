# kv_common.py -- shared helpers for the KV-decomposition stages (3 & 4).
# transformers 4.57.x cache API: cache[i] -> (K, V); cache.layers[i].keys.

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

NAME = "unsloth/Llama-3.2-1B-Instruct"
DEVICE = "cuda:0"


def load_model():
    assert torch.cuda.is_available(), "CUDA GPU required"
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(
        NAME, dtype=torch.bfloat16, device_map=DEVICE
    ).eval()
    return tok, model


_WIKI_IDS = None


def long_text_ids(tok, n_tokens, device):
    """First n_tokens of WikiText-2 test as input_ids [1, n] (varied natural text)."""
    global _WIKI_IDS
    if _WIKI_IDS is None:
        from datasets import load_dataset

        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        _WIKI_IDS = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    assert _WIKI_IDS.shape[1] >= n_tokens, "WikiText-2 test too short"
    return _WIKI_IDS[:, :n_tokens].to(device)


# ---------------- SVD helpers ----------------
def svd_compress(x, r):
    """x: [B,H,S,D] -> rank-r reconstruction of each [S,D] slice, float32."""
    b, h, s, d = x.shape
    U, sv, Vh = torch.linalg.svd(x.float().reshape(b * h, s, d), full_matrices=False)
    r = min(r, sv.shape[-1])
    recon = (U[..., :r] * sv[..., :r].unsqueeze(-2)) @ Vh[..., :r, :]
    return recon.reshape(b, h, s, d)


def svd_factors(x, r):
    """x: [B,H,S,D] -> factors A [B,H,S,r] (=U*s), Bf [B,H,r,D] (=Vh) in x.dtype,
    so that A @ Bf is the rank-r reconstruction."""
    b, h, s, d = x.shape
    U, sv, Vh = torch.linalg.svd(x.float().reshape(b * h, s, d), full_matrices=False)
    r = min(r, sv.shape[-1])
    A = (U[..., :r] * sv[..., :r].unsqueeze(-2)).reshape(b, h, s, r).to(x.dtype)
    Bf = Vh[..., :r, :].reshape(b, h, r, d).to(x.dtype)
    return A, Bf


# ---------------- caches ----------------
class LowRankCache(DynamicCache):
    """Naive/accuracy variant (handout Stage 2-3): right after prefill, replace
    the stored dense K/V with their rank-r reconstruction. Storage stays dense
    (no memory saved); the one-time SVD cost lands in TTFT."""

    def __init__(self, rank=32, compress_keys=True, compress_values=True, **kw):
        super().__init__(**kw)
        self.rank = rank
        self.ck = compress_keys
        self.cv = compress_values
        self._done = set()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx not in self._done and keys.shape[-2] > 1:
            if self.ck:
                keys = svd_compress(keys, self.rank).to(keys.dtype)
            if self.cv:
                values = svd_compress(values, self.rank).to(values.dtype)
            self.layers[layer_idx].keys = keys
            self.layers[layer_idx].values = values
            self._done.add(layer_idx)
        return keys, values


class FactoredLowRankCache(DynamicCache):
    """Memory-honest variant (handout Part V tip): the prefill K/V are stored as
    SVD factors A [S,r], B [r,D] per (layer, head); decode tokens append to an
    exact dense tail. Dense K/V are rebuilt (A @ B, concat tail) at EVERY
    update() call, so the per-step reconstruction cost is paid explicitly."""

    def __init__(self, rank=32, compress_keys=True, compress_values=True, **kw):
        super().__init__(**kw)
        self.rank = rank
        self.ck = compress_keys
        self.cv = compress_values
        self.base = {}  # layer_idx -> {"S0": int, "K": factors|dense, "V": ...}
        self.tail = {}  # layer_idx -> [k_tail, v_tail] (exact, dense)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self.base:
            assert key_states.shape[-2] > 1, "expected a multi-token prefill first"
            self.base[layer_idx] = {
                "S0": key_states.shape[-2],
                "K": svd_factors(key_states, self.rank) if self.ck else key_states,
                "V": svd_factors(value_states, self.rank) if self.cv else value_states,
            }
            self.tail[layer_idx] = [None, None]
        else:
            t = self.tail[layer_idx]
            t[0] = key_states if t[0] is None else torch.cat([t[0], key_states], dim=-2)
            t[1] = value_states if t[1] is None else torch.cat([t[1], value_states], dim=-2)
        return self._dense(layer_idx)

    def _dense(self, layer_idx):
        e = self.base[layer_idx]
        out = []
        for part, tail in zip(("K", "V"), self.tail[layer_idx]):
            p = e[part]
            base = (p[0] @ p[1]) if isinstance(p, tuple) else p  # reconstruct
            out.append(base if tail is None else torch.cat([base, tail], dim=-2))
        return out[0], out[1]

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self.base:
            return 0
        kt = self.tail[layer_idx][0]
        return self.base[layer_idx]["S0"] + (0 if kt is None else kt.shape[-2])

    def get_mask_sizes(self, cache_position, layer_idx):
        return cache_position.shape[0] + self.get_seq_length(layer_idx), 0


def kivi_quant_dequant(K, V, group=32, residual=32, bits=2):
    """KIVI-style asymmetric low-bit quantization (Liu et al., ICML 2024):
    keys per-CHANNEL (grouped along the token axis), values per-TOKEN (grouped
    along the channel axis); the most recent `residual` tokens stay full
    precision. Quantize+dequantize in one step (accuracy variant)."""
    levels = 2**bits - 1

    def qdq(x, grp_dim):
        # quantize x in groups of `group` along grp_dim (last or second-to-last)
        mn = x.min(dim=grp_dim, keepdim=True).values
        mx = x.max(dim=grp_dim, keepdim=True).values
        scale = (mx - mn).clamp_min(1e-8) / levels
        return (((x - mn) / scale).round().clamp(0, levels)) * scale + mn

    Kf, Vf = K.float(), V.float()
    B, H, S, D = Kf.shape

    # keys: per-channel, token groups of `group`, last `residual` tokens exact
    nq = max(0, ((S - residual) // group) * group)
    if nq > 0:
        g = Kf[..., :nq, :].reshape(B, H, nq // group, group, D)
        Kq = torch.cat([qdq(g, 3).reshape(B, H, nq, D), Kf[..., nq:, :]], dim=-2)
    else:
        Kq, nq = Kf, 0

    # values: per-token, channel groups of `group`, last `residual` tokens exact
    nqv = max(0, S - residual) if D % group == 0 else 0
    if nqv > 0:
        g = Vf[..., :nqv, :].reshape(B, H, nqv, D // group, group)
        Vq = torch.cat([qdq(g, 4).reshape(B, H, nqv, D), Vf[..., nqv:, :]], dim=-2)
    else:
        Vq = Vf

    # analytic compression: quantized elems cost bits + 2 fp16 stats per group
    bits_per_q = bits + 2 * 16.0 / group
    q_bits = (nq + nqv) * D * bits_per_q + ((S - nq) + (S - nqv)) * D * 16.0
    ratio = (2 * S * D * 16.0) / q_bits
    return Kq.to(K.dtype), Vq.to(V.dtype), ratio


class KIVICache(DynamicCache):
    """Compress-after-prefill cache using KIVI-style 2-bit quant-dequant
    (accuracy variant: storage stays dense, compression ratio is analytic)."""

    def __init__(self, group=32, residual=32, bits=2, **kw):
        super().__init__(**kw)
        self.group, self.residual, self.bits = group, residual, bits
        self.ratio = None
        self._done = set()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx not in self._done and keys.shape[-2] > 1:
            keys, values, self.ratio = kivi_quant_dequant(
                keys, values, self.group, self.residual, self.bits
            )
            self.layers[layer_idx].keys = keys
            self.layers[layer_idx].values = values
            self._done.add(layer_idx)
        return keys, values


def cache_stored_bytes(cache):
    """Bytes actually held by the cache's storage (factors+tails for the
    factored variant, dense tensors otherwise)."""
    if isinstance(cache, FactoredLowRankCache):
        total = 0
        for i, e in cache.base.items():
            for part in ("K", "V"):
                p = e[part]
                ts = p if isinstance(p, tuple) else (p,)
                total += sum(t.numel() * t.element_size() for t in ts)
            total += sum(
                t.numel() * t.element_size() for t in cache.tail[i] if t is not None
            )
        return total
    total = 0
    for i in range(len(cache)):
        K, V = cache[i]
        if K is not None:
            total += K.numel() * K.element_size() + V.numel() * V.element_size()
    return total


# ---------------- measurement (handout Part III) ----------------
def time_ms(fn, warmup=3, iters=10):
    """Median/P95 latency of fn() in ms via CUDA events (handout recipe)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    t = np.array(ts)
    return {"median": float(np.median(t)), "p95": float(np.percentile(t, 95))}


def energy_joules(fn, iters=4):
    """Mean Joules per fn() call via the NVML total-energy counter."""
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    fn()  # warmup
    torch.cuda.synchronize()
    mj0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    mj1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    return (mj1 - mj0) / 1000.0 / iters


def peak_mem_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()
