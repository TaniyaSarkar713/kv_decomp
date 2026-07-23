# kv_decomp — Tensor / Low-Rank / Sparse KV-Cache Compression

Research notes, a literature survey, and a proposal portfolio on compressing the
transformer **KV cache** with tensor / low-rank / sparse structure, plus the
collaborative-inference extensions this work grows into.

## Contents

**Survey & agenda**
- `KV-Cache_Low-Rank-Compression_Survey-and-Research-Agenda.{tex,pdf}` — scoped survey of ~27
  low-rank / latent / tensor / sparse-residual KV methods, a novelty-boundary analysis, the
  flagship **SAGE-CP-KV** design, and a ranked research portfolio (§12 comparison, §14 evaluation).

**Beginner handout**
- `KV-Decomposition_Beginner-Handout.{tex,pdf}` — a from-scratch guide to evaluating low-rank /
  tensor decomposition of the LLaMA-3.2 KV cache on a single device (metrics, correct
  benchmarking, staged plan, runnable code).

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
- Build with `pdflatex` (two–three passes for the table of contents, longtables, and references).
- All quantitative thresholds in the proposals are **decision gates / targets**, not measured
  results.
