# stage3_benchmark.py
# Stage 3 of the KV-Decomposition handout: does low-rank KV compression
# actually run faster/smaller, or does reconstruction overhead dominate?
#
# Variants (greedy decoding, N_NEW new tokens, WikiText-2 prompts):
#   dense       -- unmodified DynamicCache (baseline)
#   naive-rXX   -- LowRankCache: one-time SVD after prefill, storage stays dense
#                  (accuracy experiment: no memory saved, SVD cost in TTFT)
#   factored-rXX-- FactoredLowRankCache: stores SVD factors, reconstructs dense
#                  K/V every decode step (memory experiment: cache shrinks,
#                  per-step reconstruction cost added)
#
# Metrics per handout Part III: TTFT, TPOT, E2E, throughput, peak GPU memory,
# stored cache bytes (measured + analytic), energy/token (NVML).

import json
import torch
import numpy as np
from transformers import DynamicCache
from kv_common import (
    load_model, long_text_ids, LowRankCache, FactoredLowRankCache,
    cache_stored_bytes, time_ms, energy_joules, peak_mem_bytes,
)

torch.manual_seed(0)
tok, model = load_model()
cfg = model.config
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

CONTEXTS = [512, 4096]
RANKS = [16, 32]
N_NEW = 64
TTFT_ITERS, E2E_ITERS, EN_ITERS = 10, 6, 4

results = {}

for ctx in CONTEXTS:
    ids = long_text_ids(tok, ctx, model.device)

    def gen(ctor, n):
        return model.generate(
            input_ids=ids, do_sample=False, max_new_tokens=n, min_new_tokens=n,
            past_key_values=ctor(), use_cache=True, pad_token_id=tok.eos_token_id,
        )

    variants = [("dense", lambda: DynamicCache(config=cfg))]
    for r in RANKS:
        variants.append((f"naive-r{r}", lambda r=r: LowRankCache(rank=r, config=cfg)))
        variants.append((f"factored-r{r}", lambda r=r: FactoredLowRankCache(rank=r, config=cfg)))

    # correctness check: factored must reproduce the naive tokens (same SVD,
    # exact tail) -- and full-rank factored should match dense
    dense_toks = gen(lambda: DynamicCache(config=cfg), 32)[0, ctx:]
    naive_toks = gen(lambda: LowRankCache(rank=32, config=cfg), 32)[0, ctx:]
    fact_toks = gen(lambda: FactoredLowRankCache(rank=32, config=cfg), 32)[0, ctx:]
    fullrank_toks = gen(lambda: FactoredLowRankCache(rank=64, config=cfg), 32)[0, ctx:]
    agree_nf = int((naive_toks == fact_toks).sum())
    agree_fd = int((fullrank_toks == dense_toks).sum())
    print(f"\n=== context {ctx} ===")
    print(f"validation: naive-r32 vs factored-r32 tokens agree {agree_nf}/32; "
          f"factored-r64 (full rank) vs dense agree {agree_fd}/32")

    results[ctx] = {"validation": {"naive_vs_factored_r32": agree_nf, "fullrank_vs_dense": agree_fd}}
    dense_kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2 * ctx

    hdr = (f"{'variant':>13} | {'TTFT ms':>9} | {'TPOT ms':>8} | {'E2E ms':>9} | "
           f"{'tok/s':>6} | {'peak MiB':>8} | {'cache MiB':>9} | {'J/tok':>6}")
    print(hdr); print("-" * len(hdr))

    for name, ctor in variants:
        ttft = time_ms(lambda: gen(ctor, 1), warmup=2, iters=TTFT_ITERS)
        e2e = time_ms(lambda: gen(ctor, N_NEW), warmup=1, iters=E2E_ITERS)
        tpot = (e2e["median"] - ttft["median"]) / (N_NEW - 1)
        toks_s = N_NEW / (e2e["median"] / 1000.0)
        joules = energy_joules(lambda: gen(ctor, N_NEW), iters=EN_ITERS)
        peak = peak_mem_bytes(lambda: gen(ctor, N_NEW))

        c = ctor()
        with torch.no_grad():
            model(input_ids=ids, past_key_values=c, use_cache=True)
        stored = cache_stored_bytes(c)
        del c

        row = {
            "ttft_ms": ttft, "e2e_ms": e2e, "tpot_ms": tpot,
            "throughput_tok_s": toks_s, "peak_mem_bytes": int(peak),
            "cache_bytes_prefill": int(stored),
            "dense_cache_bytes_analytic": dense_kv_bytes,
            "energy_J_per_tok": joules / N_NEW,
        }
        results[ctx][name] = row
        print(f"{name:>13} | {ttft['median']:>9.1f} | {tpot:>8.2f} | "
              f"{e2e['median']:>9.1f} | {toks_s:>6.1f} | {peak/2**20:>8.0f} | "
              f"{stored/2**20:>9.2f} | {joules/N_NEW:>6.3f}")

with open("results/stage3_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved: results/stage3_results.json")
