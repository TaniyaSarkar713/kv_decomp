# kv_decomp — Tensor / Low-Rank / Sparse KV-Cache Compression

Research notes, a literature survey, a proposal portfolio, and staged experiments on
compressing the transformer **KV cache** with tensor / low-rank / sparse structure, plus
the collaborative-inference extensions this work grows into.

## Repository layout

```
├── README.md
├── RESULTS_AND_NEXT_EXPERIMENTS.md   # measured findings + gated experiment plan
├── experiments/                      # staged experiment scripts (run from repo root)
│   ├── kv_common.py                  # shared model/data/cache/measurement helpers
│   ├── stage1_kv_extract.py          # KV extraction, SVD spectra, rank-r next-token check
│   ├── stage3_benchmark.py           # dense vs reconstructed vs factored: TTFT/TPOT/energy
│   ├── stage4_sweep.py               # rank x context x K/V/KV accuracy + latency Pareto
│   ├── stage4_gsm8k.py               # GSM8K reasoning guard under cache compression
│   ├── stage5_baselines_tensor.py    # KIVI-style 2-bit, Tucker, Tensor-Train baselines
│   └── stage6_shufflerless_phase0.py # Shufflerless Phase 0 + SAGE residual + payback
│                                     #   + CP / SC-KV / FD-KV / K-only smoke tests
├── results/                          # measured JSON results + figures (committed)
├── logs/                             # run logs (gitignored)
└── references/                       # survey, handout, proposals (tex + pdf)
```

## Running the experiments

```bash
# from the repo root, with the project venv active (torch + CUDA, transformers,
# datasets, tensorly, matplotlib); outputs land in results/
python experiments/stage6_shufflerless_phase0.py
```

Model: `unsloth/Llama-3.2-1B-Instruct` (bf16, single GPU). Data: WikiText-2
(train split for calibration, test split for held-out evaluation).
Stage 6 caches its offline calibration in `results/stage6_calib.pt` (gitignored,
regenerated automatically on first run).

## Experiments (summary)

- **Stages 1–5** establish the baseline picture: keys are moderately low-rank, values are
  not; symmetric rank-only K/V compression is rejected in the useful-compression regime;
  KIVI-style 2-bit quantization is the strong quality baseline; plain `[H,S,D]`
  Tucker/TT adds no useful structure. See `RESULTS_AND_NEXT_EXPERIMENTS.md`.
- **Stage 6** (`results/stage6_results.json`, `results/stage6_phase0.png`) tests the
  proposal-portfolio mechanisms with an exact hot window (W=64) and a teacher-forced
  64-token held-out tail: (A) a **fixed, offline-calibrated latent basis** ("born
  compressed", no runtime SVD) matches the per-prompt oracle SVD; (B) the **SAGE
  attention-influence sparse residual** repairs a broken rank-16 cache from KL ≈ 1.2 to
  ≈ 0.06 with only 2% of RoPE-pair groups at equal bytes, while energy/magnitude/random
  selection barely help; (C) **payback admission**: fixed-basis block coding costs
  2.2 ms vs 614 ms for runtime SVD (breakeven ≈ 10² vs ≈ 10⁴–10⁵ future tokens);
  unfused sparse correction has negative latency savings (a fused kernel is required);
  (D) smoke tests: CP bulk (SAGE-CP) is byte-competitive with the cheapest coding and
  fastest long-context read, FD-KV streaming matches oracle SVD without calibration,
  SC-KV union-of-subspaces beats one subspace at equal bytes, and K-only compression
  loses to KV-at-higher-rank at equal bytes.

## Documents (`references/`)

**Survey & agenda**
- `KV-Cache_Low-Rank-Compression_Survey-and-Research-Agenda.{tex,pdf}` — scoped survey of ~27
  low-rank / latent / tensor / sparse-residual KV methods, a novelty-boundary analysis, the
  flagship **SAGE-CP-KV** design, and a ranked research portfolio (§12 comparison, §14 evaluation).

**Beginner handout**
- `KV-Decomposition_Beginner-Handout.{tex,pdf}` — a from-scratch guide to evaluating low-rank /
  tensor decomposition of the LLaMA-3.2 KV cache on a single device (metrics, correct
  benchmarking, staged plan, runnable code).
- `LLaMA-3.2_Local-Inference-Guide.doc` — companion local-inference guide.

**Proposals**
- `SparsePipe_Proposal.{tex,pdf}` — sparsity as a schedulable distributed resource.
- `LoSpar-CI_Proposal.{tex,pdf}` — progressive low-rank-plus-sparse tensor inference across devices.
- `Shufflerless-CI_Proposal.{tex,pdf}` — stationary latent–residual collaborative decoding.
- `CATT-KV_Proposal.{tex,pdf}` — causal Tensor-Train with sparse innovations.
- `CP-Direct-KV_Proposal.{tex,pdf}` — direct compressed-domain CP attention.

**Background briefs**
- `Collaborative-Inference_Beginner-Summary.tex`
- `Research-Agenda_Sparsity-First-Collaborative-Inference.{tex,pdf}`
- `Tensor_Low-Rank_Sparse_Inference_Research-Brief.pdf`

## Notes

- The original third-party paper PDFs (HexGen, Jupiter, and the RL/MARL/FL references) are
  **intentionally excluded** from this repository (see `.gitignore`) to avoid redistributing
  copyrighted material; consult them at their official venues / arXiv.
- Build the documents with `pdflatex` (two–three passes for the table of contents, longtables,
  and references).
- All quantitative thresholds in the proposals are **decision gates / targets**, not measured
  results; measured numbers live in `results/` and `RESULTS_AND_NEXT_EXPERIMENTS.md`.
