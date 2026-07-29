# KV-Cache Decomposition: Results and Next Experiments

**Date:** 2026-07-26  
**Scope:** Llama-3.2-1B-Instruct on one RTX 4090, together with the nine PDFs in
`references/`.

## Executive conclusion

The experiments reject **plain symmetric low-rank KV compression** as a useful
standalone direction in the tested regime. Keys have moderate low-rank structure, but
values are much closer to full rank; ranks that give attractive compression alter model
behavior, while ranks that preserve a sampled next token give only about 2x compression.
Ordinary Tucker and Tensor-Train decompositions over `[head, token, feature]` do not
improve this frontier. Reconstructing dense KV from factors also removes most of the
systems benefit.

The best next direction is therefore an **asymmetric, attention-aware old-cache
hybrid**:

1. keep a recent and protected KV region exact;
2. use a structured or latent representation primarily where it helps keys;
3. quantize values unless a separate value representation passes its own gate;
4. retain a small, RoPE-pair-aligned sparse residual selected by attention influence;
5. consume the old representation directly, without reconstructing dense KV; and
6. fall back per block to packed INT4/KIVI-style or native KV when compression will
   not repay its cost.

This is closest to a narrowed **Shufflerless-CI Phase 0**, using the residual design of
**SAGE-CP-KV/CP-Direct-KV** and the admission rule of **Payback-KV**. It is a hypothesis
to falsify cheaply, not yet a validated system.

## 1. Evidence and status

The measured artifacts are:

- [`stage1_results.json`](results/stage1_results.json): one 441-token varied prompt; per-head
  post-RoPE spectra, reconstruction error, and a next-token check.
- [`stage3_results.json`](results/stage3_results.json): dense, dense-reconstructed, and
  factor-stored SVD prototypes at contexts 512 and 4096.
- [`stage4_results.json`](results/stage4_results.json): rank/context/K-vs-V sweep and a small
  latency sweep.
- [`stage5_results.json`](results/stage5_results.json): KIVI-style quantize/dequantize,
  per-head SVD, Tucker, and TT.
- [`gsm8k_partial.json`](results/gsm8k_partial.json): completed full-cache and rank-48
  configurations on 50 GSM8K problems; later configurations were interrupted.

The numerical targets in the proposal PDFs are **decision gates, not achieved
results**. No current experiment implements CP factors, influence-selected sparse
residuals, packed KIVI storage, a direct compressed-domain kernel, or a distributed
pipeline.

## 2. Results

### 2.1 Structural premise

The cache has 16 layers and per-layer K/V shape `[1, 8, S, 64]`, read as
`[batch, kv_heads, sequence_length, head_dim]`. **S is the sequence length**: the
number of tokens cached so far (prompt plus generated tokens). It is the only
dimension that grows during decoding, by one per token. The 8 KV heads come from
grouped-query attention (32 query heads share 8 KV heads).

Per-token memory in bf16 (2 bytes/element): one layer stores
8 heads x 64 dims x 2 B = 1 KiB for K and the same for V, so 2 KiB per layer,
and 16 layers x 2 KiB = **32 KiB per token across all layers**. The total cache
is therefore `S x 32 KiB`: 16 MiB at S = 512, 64 MiB at 2,048, 128 MiB at 4,096,
256 MiB at 8,192, and 4 GiB at the model's 131,072-token context limit — more
than the 1B model's own weights (~2.5 GiB in bf16), which is the memory-wall
motivation for compressing the cache. (Cross-check: the stage-3 table below
measures 16 MiB dense at context 512 and 128 MiB at 4,096.)

| Quantity | Keys | Values |
|---|---:|---:|
| Mean rank for 90% energy | 20.75 | 40.25 |
| Mean rank for 95% energy | 30.06 | 49.44 |
| Mean rank for 99% energy | 49.19 | 60.31 |

Values are therefore only weakly low-rank. The prompt-specific SVD reconstruction
results reinforce this:

| Rank | K relative error | V relative error | Analytic factor compression |
|---:|---:|---:|---:|
| 4 | 0.560 | 0.811 | 13.97x |
| 8 | 0.483 | 0.724 | 6.99x |
| 16 | 0.369 | 0.595 | 3.49x |
| 32 | 0.205 | 0.401 | 1.75x |

**Interpretation:** there is exploitable structure, especially in K, but not enough to
justify one symmetric rank for both K and V. Spectral energy is only a diagnostic;
behavior must determine the operating point.

### 2.2 Behavioral frontier

For jointly compressed K and V:

| Context | Rank | Compression | KL(full‖compressed) | Top-1 preserved |
|---:|---:|---:|---:|:---:|
| 512 | 16 | 3.55x | 7.630 | No |
| 512 | 32 | 1.78x | 0.574 | Yes |
| 2048 | 16 | 3.88x | 4.929 | No |
| 2048 | 32 | 1.94x | 0.698 | Yes |
| 8192 | 16 | 3.97x | 7.827 | No |
| 8192 | 32 | 1.98x | 1.070 | Yes |

K and V are behaviorally asymmetric as well as spectrally asymmetric. At context 2048,
V-only rank 8 preserves the sampled top token with KL 1.282, whereas K-only rank 8 changes
it with KL 6.546. K has lower Frobenius error but is more capable of changing attention
routing. This is direct evidence against choosing ranks from reconstruction error alone.

Top-1 agreement at one position is also too weak a quality criterion: rank 32 can preserve
the token while retaining KL as high as 1.07.

### 2.3 Memory, latency, and energy

The factor-stored prototype saves cache bytes but reconstructs dense old K/V on every
decode update.

| Context | Variant | Stored prefill KV | TTFT (ms) | TPOT (ms) | Energy (J/token) |
|---:|---|---:|---:|---:|---:|
| 512 | Dense | 16.0 MiB | 28.9 | 21.24 | 1.989 |
| 512 | Factored r16 | 4.5 MiB | 627.1 | 24.70 | 3.565 |
| 512 | Factored r32 | 9.0 MiB | 607.2 | 28.91 | 3.689 |
| 4096 | Dense | 128.0 MiB | 527.5 | 21.04 | 6.339 |
| 4096 | Factored r16 | 32.5 MiB | 996.6 | 24.29 | 7.289 |
| 4096 | Factored r32 | 65.0 MiB | 970.9 | 22.04 | 6.904 |

At context 4096, rank 16 stores 3.94x less KV but increases TTFT by 89%, TPOT by
15.5%, and energy/token by 15%. This refutes the **reconstruct-every-step
implementation**, not direct compressed-domain attention. It motivates born-compressed
representations, block aging, direct attention, and explicit payback admission.

The factored path is also not numerically equivalent to the dense-reconstructed path:
rank-32 continuations agree on only 5/32 tokens at context 512, while even full-rank
factors agree with dense on only 29/32 tokens at context 4096. Factor precision and
operation order therefore need to be controlled before combining quality from one path
with latency from another. The timing sweeps use only 3–10 measured iterations, so their
non-monotone TPOT points are preliminary rather than a stable kernel crossover.

### 2.4 Quantization and multiway tensor baselines

The KIVI-style quality baseline is substantially stronger than rank reduction on the
sampled next-token test:

| Context | Analytic compression | KL | Top-1 preserved |
|---:|---:|---:|:---:|
| 512 | 3.80x | 0.991 | Yes |
| 2048 | 4.85x | 0.215 | No |
| 8192 | 5.20x | 0.064 | Yes |

This implementation quantizes and then materializes dense tensors; its compression is
analytic. It establishes a strong **representation-quality baseline**, not a physical
memory or speed result.

At context 2048, the gentlest Tucker and TT settings give only 1.77x and 1.99x
compression, respectively, yet KL is 4.65–4.74 and both change the top token. Per-head SVD
at a comparable 1.94x gives KL 0.698 and preserves it. The tested cross-head tensorization
therefore adds no useful structure. It does not directly test semantic CP or CATT's dyadic
causal TT, but it removes any reason to prioritize a large tensor implementation before a
cheap rank pilot.

### 2.5 Multi-step reasoning

The completed GSM8K configurations are:

| Cache | Correct | Accuracy | Mean generated tokens |
|---|---:|---:|---:|
| Full | 19/50 | 38% | 148 |
| Joint KV rank 48 | 0/50 | 0% | 137 |

The rank-48 run frequently degenerates to repeated generic answers such as 42 and often
reaches the token cap. There are no parser failures. Thus the result is a serious warning
that small per-step errors compound during generation.

It is not a clean test of an aged-block hybrid: the current cache hook compresses and
returns K/V during the multi-token prefill, thereby changing prompt self-attention, and it
has no exact hot or protected region. Only two configurations completed. The result should
therefore motivate a corrected downstream experiment, not be presented as a definitive
failure of SAGE, CP-Direct, or Shufflerless.

## 3. Conclusions supported by the evidence

1. **Do not continue symmetric rank-only K/V as the primary method.**
2. **Treat K and V differently.** K is more spectrally compressible but more
   routing-sensitive; V is less low-rank and is a natural quantization candidate.
3. **Do not optimize Frobenius error alone.** Use query/attention-output error and
   multi-token tasks.
4. **Do not reconstruct dense old KV per token.** A useful representation must be directly
   executable or justified purely as a capacity method.
5. **INT4/KIVI-style quantization is the mandatory baseline and likely fallback.**
6. **Exact recent/protected storage plus sparse innovations is the most plausible quality
   rescue**, but residual concentration has not yet been measured.
7. **Ordinary Tucker/TT is not promising in the tested layout.**
8. **No distributed project is justified until a single-device codec produces a real
   quality–physical-memory–latency frontier.**

## 4. Next experiments

The experiments should follow this dependency chain:

```text
repair/validate harness
        ↓
packed baseline frontier
        ↓
asymmetric K/V + exact-hot + sparse-residual quality pilot
        ↓ passes
CP/latent structural pilot → direct-kernel crossover → single-GPU integration
        ↓ passes
stationary multi-device Shufflerless-CI

Any failed quality/physical-byte gate → quantization/eviction pivot, not more tensor complexity
```

### Experiment 0 — Repair and validate the harness

**Question:** Are quality differences caused by the intended cache approximation rather
than prefill semantics, precision, or cache implementation?

**Protocol**

- Produce dense prefill K/V first; compress only closed old blocks afterward.
- Add an explicit exact hot window.
- On the same GSM8K examples, compare the current in-prefill hook against dense prefill
  followed by old-block compression. Include `W = S` (the entire prompt exact) as the
  identity endpoint. This isolates codec damage from the current prefill-hook behavior.
- Make the full-rank reconstructed, factor-stored, and dense paths use matched precision.
- Compare logits, attention outputs, and 32-token greedy continuations.
- Add the missing dense perplexity baseline.
- Record direct per-token CUDA-event times, including block-aging spikes; do not derive
  TPOT by subtracting independently measured TTFT and E2E medians.
- Randomize/interleave variant order across repeated trials and report P50/P95/P99
  inter-token latency plus incremental peak/workspace memory. Collect at least 1,000
  post-warmup token intervals per condition before interpreting P99.

**Gate**

- An unmodified cache-copy control must reproduce the dense 32-token continuation exactly.
- FP32 direct-versus-explicit execution of the same factors should have relative error
  below `1e-4` and identical greedy tokens.
- Full-rank factors versus dense KV should have next-logit KL below `1e-3`; any later
  autoregressive divergence from rounding must be measured separately rather than mistaken
  for an implementation error.
- If this fails, stop: no later quality or latency comparison is interpretable.

### Experiment 1 — Establish a physical packed baseline frontier

**Question:** What is the real quality–memory–latency frontier before proposing a new
representation?

**Methods**

- Dense BF16/FP16.
- Actually packed INT8, INT4, and KIVI-style asymmetric quantization.
- Per-head SVD for K-only, V-only, and K+V.
- Exact-window variants with `W ∈ {64, 128, 256, 512}`.

**Evaluation**

- Contexts 512, 2K, 8K, and the longest valid context.
- Perplexity; many-position KL/attention-output error; at least one retrieval task and
  one multi-step reasoning task.
- Physical cache bytes including scales, indices, padding, and exact window.
- TTFT, TPOT P50/P95, throughput, energy, and bytes read per output token.
- Start on the 1B model, then repeat the surviving points on the 3B model and a second
  architecture family.

**Gate**

- Retire plain low-rank as a deployment candidate if no point achieves at least 2x
  physical KV reduction with at most one absolute percentage point average task loss
  (or a pre-registered statistically equivalent tolerance).

### Experiment 2 — Test asymmetric K/V representations

**Hypothesis:** structured K plus quantized V gives a better same-byte frontier than
symmetric SVD.

Run an equal-physical-byte grid:

- K: native, per-head SVD/fixed latent, INT4/KIVI-style.
- V: native, per-head SVD, INT4/KIVI-style.
- Optional per-layer/head rank and precision allocation.

Measure attention-score error separately from value-aggregation error, then full
attention-output error and downstream quality.

**Gate**

- Continue only if an asymmetric point Pareto-dominates symmetric SVD and is competitive
  with packed KIVI/INT4 at the same physical bytes across more than one context and task.

### Experiment 3 — Exact hot window and attention-aware sparse innovations

**Hypothesis:** a small structured residual repairs the routing/reasoning failures that
rank or quantization alone misses.

Starting from the best Experiment 2 baseline:

- compress only blocks after they leave the exact window;
- store residuals as whole RoPE-pair or token-by-RoPE-pair groups;
- compare equal-byte selectors: random, magnitude, energy, a causal query/influence
  score, and observed future-query attention as an explicitly oracle-only upper bound;
- sweep exact-window and residual budgets;
- separately ablate protection for instructions, delimiters, numbers, and named entities.

Use GSM8K, a retrieval benchmark, instruction-following tests, many-position KL, and
attention-output error. Report the fraction of approximation error and task loss recovered
per residual byte. Fit query envelopes and past-attention statistics only on calibration
prompts or earlier tokens, then evaluate residual selection on held-out future queries and
domains. Count residual indices and alignment in every equal-byte comparison.

**Gate**

- On held-out queries, the causal influence selector must reduce median attention-output
  error by at least 15% versus the best causal non-oracle selector, not worsen P95 error,
  and win at least 70% of layer-by-domain strata.
- A first system candidate must achieve at least 2x physical reduction with near-full
  task quality. Advancing the SAGE/CP sparse route requires residual density at or below
  10% of RoPE-pair groups while the complete format remains at least 4x smaller.
- If the residual is not sparse enough, stop the tensor+sparse direction.

### Experiment 4 — Offline CP/latent feasibility pilot

**Question:** Does semantic CP provide structure that ordinary `[H,S,D]` Tucker/TT missed?

Record K, V, and representative queries from two models and three domains. For aged blocks
of 128–512 tokens, compare:

- joint-head and head-private CP over
  `[token, head, RoPE-pair, pair-component]`;
- per-head SVD and fixed weight-side latent bases;
- pre- versus post-RoPE representations;
- CP-K plus quantized-V;
- CP-only versus CP plus the winning residual from Experiment 3.

Report rank/error curves, Gram conditioning, fit time, actual factor/residual bytes, and
attention-output quality. Verify the direct CP equations against explicit reconstruction.

**Gate**

- Do not write a GPU kernel unless CP is stable, the direct equations agree below `1e-4`
  in FP32, and the full representation—including fixed factors, metadata, hot window, and
  residual—shows a credible same-byte improvement over packed INT4/KIVI.
- CP should pass at least 60% of layer-blocks in every tested model/domain at rank 32 or
  below, with five-restart reconstruction/error coefficient of variation at or below 10%
  and no numerical failures.
- If only K passes, pivot immediately to CP-K plus packed quantized V rather than forcing
  or wholly rejecting CP.
- Stop CP if useful ranks or residual bytes eliminate the targeted 4x physical reduction.

### Experiment 5 — Direct-attention crossover microbenchmark

Implement only the representation that passed Experiment 4. Compare:

- explicit dense reconstruction;
- direct factor/latent attention;
- dense FlashAttention;
- the best packed INT4/KIVI kernel.

Sweep context/block length, batch, rank, head type, factor precision, hot-window size, and
residual density. Measure allocation, HBM bytes, kernel latency, and block-coding time.

**Gate**

- No old dense-KV allocation.
- At quality-passing ranks, at least 1.25x attention speedup over dense and preferably
  at least 1.10x over the packed low-bit kernel by context 16K. Otherwise label the method
  capacity-only rather than a speedup.
- Positive payback within the predicted block lifetime; reject a block if its payback
  horizon exceeds 25% of the expected remaining generation. Target median payback at or
  below 64 future reads and P90 at or below 128; reject the block if INT4 is faster and no
  larger.

Only after this gate should a fused Triton/CUDA implementation be expanded.

### Experiment 6 — Integrated single-device generation

Integrate exact-hot, compressed-old, sparse residual, blockwise softmax merge, and per-block
native/INT4 fallback. Repeat the complete baseline suite on two model families.

The integrated claim requires measured physical bytes, fit/aging cost, TTFT, P50/P95 TPOT,
energy, maximum context, perplexity, retrieval, reasoning, and instruction preservation.

**Gate**

- At least 2x physical KV reduction.
- A one-sided paired 95% confidence interval must exclude more than one percentage point
  aggregate task loss; no benchmark point estimate may fall by more than two points, and
  perplexity may increase by no more than 2%. Insufficient sample size leaves the gate
  unpassed rather than establishing equivalence.
- Require either at least 10% end-to-end TPOT improvement or at least 2x longer admitted
  context at matched quality.
- The controller must reject unprofitable/non-compressible blocks rather than hiding them.

### Experiment 7 — Distributed work, only after Experiment 6

If the single-device codec passes, implement the stationary part of Shufflerless-CI:

- every layer and its KV remain on one owner stage;
- jointly choose contiguous layer placement, hot-window/rank/residual budgets, and workspace;
- compare the same codec under a fixed split versus joint placement;
- trace copies, migrations, page faults, synchronization, network bytes, and P95 latency.

Target zero large KV movement for admitted requests and either 1.5x end-to-end throughput or
2–4x longer admitted context at matched quality.

SparsePipe should wait until at least two codecs have measured, distinct operating regions.
LoSpar-CI should be evaluated separately on a genuinely high-dimensional boundary tensor
such as hyperspectral or medical data. CATT-KV should proceed beyond a one-day dyadic-QTT
rank check only if useful causal ranks are near the direct-compute break-even region
(`r ≲ sqrt(64) ≈ 8` for the present head dimension).

## Final recommendation

Run Experiments 0–3 next. They are sufficient to decide whether the core hypothesis—exact
hot/protected KV plus asymmetric compression and sparse attention-aware corrections—is
real. Do not begin a CP/CATT kernel, Jetson integration, dynamic partitioner, or RL
controller until those experiments demonstrate a downstream-quality improvement at matched
physical bytes.
