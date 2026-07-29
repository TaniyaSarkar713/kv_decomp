# Live Llama-3.2 K/V and Low-Rank Demonstration

Run the demo from this repository with:

```powershell
.\.venv\Scripts\python.exe .\llama32_kv_live_demo.py --local-files-only
```

It loads the locally cached `unsloth/Llama-3.2-1B-Instruct`, uses 256 diverse
prompt tokens, prints the model and cache anatomy, analyzes all 16 layers, compares
per-head matrix SVD, flattened matrix SVD, and joint Tucker/HOSVD at matched storage,
saves three figures, and opens them interactively.

If the model is not cached, omit `--local-files-only` to allow Transformers to download
it:

```powershell
.\.venv\Scripts\python.exe .\llama32_kv_live_demo.py
```

For a non-interactive run:

```powershell
.\.venv\Scripts\python.exe .\llama32_kv_live_demo.py --no-show --local-files-only
```

For your own text:

```powershell
.\.venv\Scripts\python.exe .\llama32_kv_live_demo.py `
  --text-file .\my_prompt.txt `
  --tokens 512 `
  --layer 8 `
  --head 0
```

## What the code shows

Llama-3.2-1B has:

- 16 decoder layers;
- hidden width 2048;
- 32 query heads;
- 8 key/value heads;
- head dimension 64.

It uses Grouped-Query Attention, so four query heads share one key/value head. For each
layer, Transformers returns:

```text
K: [batch, 8 KV heads, sequence tokens, 64 head features]
V: [batch, 8 KV heads, sequence tokens, 64 head features]
```

The cached keys are post-RoPE in this implementation. Values are not rotated.

### Matrix decomposition

The script treats every KV head independently as a matrix:

```text
[tokens, 64] ≈ [tokens, r] @ [r, 64]
```

This is truncated SVD. Across all heads it stores:

```text
8 × r × (tokens + 64) scalars
```

The demo also shows a second matrix view by reshaping the same tensor to:

```text
[tokens, 8 × 64]
```

and applying one SVD. This shares a token-position basis across every head and feature,
but merges the meaning of the head and feature modes. It is an important reminder that
"matrix decomposition" depends on how a tensor is flattened.

### Tensor decomposition

The script keeps the layer cache as one three-way tensor:

```text
[8 heads, tokens, 64 features]
```

Tucker/HOSVD approximates it using a small core and three factor matrices:

```text
core[r_H, r_S, r_D]
head_factor[8, r_H]
token_factor[tokens, r_S]
feature_factor[64, r_D]
```

It can therefore share structure across heads as well as across tokens and features. This
is a stronger assumption than independent per-head SVD, and it is not automatically
better.

The script chooses Tucker ranks under the same scalar-storage budget as each SVD point.
That is the fair comparison: an SVD rank of 16 and a Tucker rank tuple such as
`(8, 32, 24)` do not mean comparable memory on their own.

## Generated outputs

The default output directory is `kv_demo_outputs/`:

- `01_kv_low_rank_overview.png` — mean energy curves, rank by layer, and a selected
  head's singular values.
- `02_matrix_vs_tensor.png` — reconstruction error versus compression for per-head
  SVD, flattened SVD, and Tucker at matched storage.
- `03_reconstruction_heatmaps.png` — original and reconstructed K/V matrices.
- `demo_results.json` — all architecture, rank, storage, and error numbers.

## How to interpret the live result

- A spectrum that reaches 95% near rank 8 out of 64 is strongly low rank.
- Reaching 95% near rank 32 is moderately low rank.
- Reaching 95% near rank 50–60 is only weakly low rank.
- If flattened SVD beats per-head SVD, one token-position basis can usefully be shared
  across heads.
- If Tucker beats flattened SVD, keeping head, token, and feature modes separate provides
  additional value at that budget.
- If per-head SVD wins, head-specific bases are more useful than the tested sharing.
- Inspect Tucker's three chosen ranks. A head rank of 8 means it kept every head
  direction; a feature rank of 64 means it kept every feature direction. In that case,
  most of its compression comes from a shared low-rank token mode rather than from
  compressing every tensor mode.

This demonstration measures structure and reconstruction. It does **not** establish that
a compressed cache preserves generation quality or improves latency. Those require a
correct cache integration, downstream tasks, physical packed storage, and a direct
compressed-domain attention kernel.
