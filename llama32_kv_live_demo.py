"""
Beginner-friendly live demonstration of Llama-3.2 KV-cache structure.

What this script demonstrates
-----------------------------
1. The compact anatomy of Llama-3.2-1B attention and Grouped-Query Attention.
2. The real K/V cache tensors produced by a prompt.
3. Whether each layer/head is approximately low rank.
4. The difference between:
   - matrix decomposition: an independent SVD of every [token, feature] head;
   - matrix decomposition after flattening heads/features into [token, head*feature];
   - tensor decomposition: one Tucker/HOSVD model of [head, token, feature].
5. A fair comparison at matched storage, not merely matched rank names.

This is a structural demonstration. Low reconstruction error does NOT by itself prove
that compressed generation will preserve perplexity, reasoning, or instructions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Diverse, non-repeated text is preferable to repeating one sentence, which can make the
# cache look artificially low rank.
DEFAULT_TEXT = """
General relativity describes gravity as the curvature of spacetime. A planet follows a
geodesic around a star, and light bends when it passes a massive object. Satellite clocks
need corrections from both special and general relativity; otherwise navigation errors
would accumulate rapidly.

Fermentation is a very different process. Yeast metabolizes sugars and releases carbon
dioxide, while lactic-acid bacteria create the sour flavor of many breads and vegetables.
Cheese, soy sauce, kimchi, coffee, and chocolate each depend on carefully controlled
microbial communities.

Before satellite navigation, sailors estimated latitude with celestial observations.
Longitude was harder because it required an accurate clock. The marine chronometer made it
possible to compare local noon with a reference time and transformed ocean navigation.

A hash table stores key-value pairs by mapping a key to a bucket. Collisions can be handled
with chaining or open addressing. A high load factor saves memory but increases the number
of probes, so practical systems resize before the table becomes crowded.

Photosynthesis captures light energy and uses it to build chemical bonds. Plants exchange
carbon dioxide and water with their environment, and chlorophyll absorbs some wavelengths
more strongly than others. The global carbon cycle connects forests, soils, oceans, and the
atmosphere over timescales ranging from days to millennia.

In music, rhythm organizes events in time while harmony describes relationships among
pitches. A performer can preserve the same written notes yet change the character of a
piece through articulation, dynamics, timing, and tone color. Musical structure therefore
contains both repeated patterns and rare expressive events.

Databases use indexes to avoid scanning every row for every query. A B-tree keeps keys in
sorted blocks and supports range searches, whereas a hash index is usually better for exact
lookups. Query optimizers estimate the cost of alternative plans using table statistics,
available memory, and expected data movement.

Weather prediction combines physical equations with observations from satellites, radar,
weather stations, balloons, and ocean buoys. Small errors in an initial state can grow over
time, which is why forecasts are often produced as ensembles rather than as one supposedly
perfect trajectory.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect real Llama-3.2 K/V tensors and compare SVD with Tucker/HOSVD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        help="Maximum number of prompt tokens. Use at least 64 for meaningful rank curves.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file. The built-in diverse text is used otherwise.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer used for the detailed decomposition; -1 selects the middle layer.",
    )
    parser.add_argument("--head", type=int, default=0, help="KV head shown in heatmaps.")
    parser.add_argument(
        "--ranks",
        default="4,8,16,32,48",
        help="Comma-separated per-head SVD ranks used for matched-budget comparisons.",
    )
    parser.add_argument(
        "--compare-rank",
        type=int,
        default=16,
        help="SVD rank whose matched Tucker reconstruction is shown in the heatmap.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kv_demo_outputs"),
        help="Directory for plots and the JSON summary.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download anything; require the model to exist in the local HF cache.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening interactive plot windows.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def load_prompt(args: argparse.Namespace) -> str:
    if args.text_file is None:
        return DEFAULT_TEXT
    return args.text_file.read_text(encoding="utf-8")


def parse_ranks(text: str, maximum: int) -> list[int]:
    try:
        values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--ranks must be comma-separated integers, for example 4,8,16") from exc
    values = [rank for rank in values if 0 < rank <= maximum]
    if not values:
        raise ValueError(f"No requested rank is between 1 and the maximum rank {maximum}.")
    return values


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def human_bytes(n_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(n_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def rank_for_energy(cumulative_energy: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return one-based ranks for curves shaped [..., rank]."""
    return (cumulative_energy < fraction).sum(dim=-1) + 1


def batched_spectrum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: [H, S, D].
    Return singular values and cumulative squared-energy curves for every head.
    """
    singular_values = torch.linalg.svdvals(x.float())
    squared = singular_values.square()
    cumulative = squared.cumsum(dim=-1) / squared.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return singular_values, cumulative


def analyze_all_layers(
    cache: Any,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Calculate per-layer/head spectra for all cached K and V tensors."""
    rows: list[dict[str, Any]] = []
    all_k_curves: list[torch.Tensor] = []
    all_v_curves: list[torch.Tensor] = []

    for layer_idx in range(len(cache)):
        key, value = cache[layer_idx]
        key_curves = batched_spectrum(key[0])[1]
        value_curves = batched_spectrum(value[0])[1]
        all_k_curves.append(key_curves.detach().cpu())
        all_v_curves.append(value_curves.detach().cpu())

        row: dict[str, Any] = {"layer": layer_idx}
        for label, curves in (("K", key_curves), ("V", value_curves)):
            for percentage, fraction in ((90, 0.90), (95, 0.95), (99, 0.99)):
                ranks = rank_for_energy(curves, fraction).float()
                row[f"{label}{percentage}_mean"] = float(ranks.mean().item())
                row[f"{label}{percentage}_min"] = int(ranks.min().item())
                row[f"{label}{percentage}_max"] = int(ranks.max().item())
        rows.append(row)

    return rows, torch.stack(all_k_curves), torch.stack(all_v_curves)


def print_layer_rank_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'layer':>5} | {'K@90':>6} {'K@95':>6} {'K@99':>6} | "
        f"{'V@90':>6} {'V@95':>6} {'V@99':>6}"
    )
    print("-" * 61)
    for row in rows:
        print(
            f"{row['layer']:>5} | "
            f"{row['K90_mean']:>6.1f} {row['K95_mean']:>6.1f} {row['K99_mean']:>6.1f} | "
            f"{row['V90_mean']:>6.1f} {row['V95_mean']:>6.1f} {row['V99_mean']:>6.1f}"
        )

    mean_k95 = sum(row["K95_mean"] for row in rows) / len(rows)
    mean_v95 = sum(row["V95_mean"] for row in rows) / len(rows)
    print(
        f"\nAcross layers, mean rank@95%: K={mean_k95:.1f}, V={mean_v95:.1f}. "
        "Smaller relative to the full rank means stronger low-rank structure."
    )


def per_head_svd_analysis(
    x: torch.Tensor, ranks: list[int]
) -> tuple[list[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Independent matrix SVD for every head.

    Each [S,D] head stores U*Sigma [S,r] and Vh [r,D], for H*r*(S+D)
    scalars across all heads.
    """
    h, s, d = x.shape
    u, singular_values, vh = torch.linalg.svd(x.float(), full_matrices=False)
    squared = singular_values.square()
    total_energy = squared.sum()
    dense_scalars = h * s * d

    rows = []
    for rank in ranks:
        retained = squared[:, :rank].sum()
        relative_error = math.sqrt(max(0.0, 1.0 - float((retained / total_energy).item())))
        stored_scalars = h * rank * (s + d)
        rows.append(
            {
                "rank": rank,
                "stored_scalars": stored_scalars,
                "compression": dense_scalars / stored_scalars,
                "relative_error": relative_error,
            }
        )
    return rows, (u, singular_values, vh)


def svd_reconstruct(
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], rank: int
) -> torch.Tensor:
    u, singular_values, vh = factors
    return (u[:, :, :rank] * singular_values[:, None, :rank]) @ vh[:, :rank, :]


def flattened_svd_analysis(
    x: torch.Tensor, budget_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Reshape [H,S,D] to one matrix [S,H*D] and use one shared token basis.

    This is a useful bridge: a matrix decomposition can also couple heads if we flatten
    dimensions first, but it no longer keeps head and feature modes explicit.
    """
    h, s, d = x.shape
    matrix = x.permute(1, 0, 2).reshape(s, h * d).float()
    singular_values = torch.linalg.svdvals(matrix)
    squared = singular_values.square()
    total_energy = squared.sum().clamp_min(1e-30)
    maximum_rank = singular_values.shape[0]
    dense_scalars = h * s * d
    factor_cost_per_rank = s + h * d

    rows = []
    for budget_row in budget_rows:
        budget = int(budget_row["stored_scalars"])
        rank = min(maximum_rank, budget // factor_cost_per_rank)
        if rank < 1:
            raise RuntimeError("The comparison budget cannot store flattened SVD rank 1.")
        stored_scalars = rank * factor_cost_per_rank
        retained = squared[:rank].sum()
        relative_error = math.sqrt(
            max(0.0, 1.0 - float((retained / total_energy).item()))
        )
        rows.append(
            {
                "matched_to_svd_rank": int(budget_row["rank"]),
                "rank": int(rank),
                "stored_scalars": int(stored_scalars),
                "compression": dense_scalars / stored_scalars,
                "relative_error": relative_error,
            }
        )
    return rows


def mode_bases_and_core(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Higher-Order SVD (HOSVD) of X[head, token, feature].

    U_h captures directions across heads.
    U_s captures directions across token positions.
    U_d captures directions across head features.
    The core records how those three sets of directions interact.
    """
    h, s, d = x.shape
    x = x.float()

    # Left singular vectors of each mode unfolding.
    u_h = torch.linalg.svd(x.reshape(h, s * d), full_matrices=False).U
    u_s = torch.linalg.svd(
        x.permute(1, 0, 2).reshape(s, h * d), full_matrices=False
    ).U
    u_d = torch.linalg.svd(
        x.permute(2, 0, 1).reshape(d, h * s), full_matrices=False
    ).U

    # Project X into the orthonormal mode bases without constructing a huge Kronecker matrix.
    core = torch.einsum("hsd,ha->asd", x, u_h)
    core = torch.einsum("asd,sb->abd", core, u_s)
    core = torch.einsum("abd,dc->abc", core, u_d)
    return u_h, u_s, u_d, core


def tucker_storage(shape: tuple[int, int, int], ranks: tuple[int, int, int]) -> int:
    """Core plus three factor matrices."""
    h, s, d = shape
    r_h, r_s, r_d = ranks
    return r_h * r_s * r_d + h * r_h + s * r_s + d * r_d


def matched_tucker_points(
    x: torch.Tensor,
    matrix_rows: list[dict[str, Any]],
    core: torch.Tensor,
) -> list[dict[str, Any]]:
    """
    For every SVD storage budget, choose the truncated HOSVD ranks with the most
    retained core energy while using no more scalars than that budget.

    This makes the comparison fair: "rank 16 SVD" and "Tucker rank (...)" do not
    mean the same thing, so they must be compared by actual stored scalar count.
    """
    h, s, d = x.shape
    max_r_h, max_r_s, max_r_d = core.shape
    dense_scalars = h * s * d

    prefix_energy = core.square().cumsum(0).cumsum(1).cumsum(2)
    total_energy = x.float().square().sum().clamp_min(1e-30)

    r_h_grid = torch.arange(1, max_r_h + 1, device=x.device).view(-1, 1, 1)
    r_s_grid = torch.arange(1, max_r_s + 1, device=x.device).view(1, -1, 1)
    r_d_grid = torch.arange(1, max_r_d + 1, device=x.device).view(1, 1, -1)
    storage_grid = (
        r_h_grid * r_s_grid * r_d_grid
        + h * r_h_grid
        + s * r_s_grid
        + d * r_d_grid
    )

    rows = []
    for matrix_row in matrix_rows:
        budget = int(matrix_row["stored_scalars"])
        valid = storage_grid <= budget
        if not bool(valid.any()):
            raise RuntimeError("The matched Tucker budget is smaller than the minimum model.")

        retained_grid = torch.where(
            valid, prefix_energy, torch.full_like(prefix_energy, -1.0)
        )
        flat_index = int(retained_grid.reshape(-1).argmax().item())
        r_d = flat_index % max_r_d + 1
        remaining = flat_index // max_r_d
        r_s = remaining % max_r_s + 1
        r_h = remaining // max_r_s + 1
        ranks = (r_h, r_s, r_d)

        retained = prefix_energy[r_h - 1, r_s - 1, r_d - 1]
        relative_error = math.sqrt(
            max(0.0, 1.0 - float((retained / total_energy).item()))
        )
        stored_scalars = tucker_storage((h, s, d), ranks)
        rows.append(
            {
                "matched_to_svd_rank": int(matrix_row["rank"]),
                "ranks": list(ranks),
                "stored_scalars": stored_scalars,
                "compression": dense_scalars / stored_scalars,
                "relative_error": relative_error,
            }
        )
    return rows


def tucker_reconstruct(
    bases_and_core: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ranks: tuple[int, int, int],
) -> torch.Tensor:
    u_h, u_s, u_d, full_core = bases_and_core
    r_h, r_s, r_d = ranks
    core = full_core[:r_h, :r_s, :r_d]
    reconstructed = torch.einsum("abc,ha->hbc", core, u_h[:, :r_h])
    reconstructed = torch.einsum("hbc,sb->hsc", reconstructed, u_s[:, :r_s])
    reconstructed = torch.einsum("hsc,dc->hsd", reconstructed, u_d[:, :r_d])
    return reconstructed


def print_comparison(
    label: str,
    matrix_rows: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    tucker_rows: list[dict[str, Any]],
) -> None:
    print(f"\n{label} tensor")
    print(
        f"{'head r':>6} | {'head comp':>9} {'head err':>8} | "
        f"{'flat r':>6} {'flat comp':>9} {'flat err':>8} | "
        f"{'Tucker ranks':>15} {'T comp':>8} {'T err':>8}"
    )
    print("-" * 106)
    for matrix_row, flat_row, tucker_row in zip(matrix_rows, flat_rows, tucker_rows):
        rank_text = "(" + ",".join(str(value) for value in tucker_row["ranks"]) + ")"
        print(
            f"{matrix_row['rank']:>6} | "
            f"{matrix_row['compression']:>8.2f}x {matrix_row['relative_error']:>8.4f} | "
            f"{flat_row['rank']:>6} {flat_row['compression']:>8.2f}x "
            f"{flat_row['relative_error']:>8.4f} | "
            f"{rank_text:>15} {tucker_row['compression']:>7.2f}x "
            f"{tucker_row['relative_error']:>8.4f}"
        )


def plot_low_rank_overview(
    rows: list[dict[str, Any]],
    all_k_curves: torch.Tensor,
    all_v_curves: torch.Tensor,
    selected_k: torch.Tensor,
    selected_v: torch.Tensor,
    layer: int,
    head: int,
    output_path: Path,
) -> Any:
    import matplotlib.pyplot as plt

    rank_axis = torch.arange(1, all_k_curves.shape[-1] + 1)
    mean_k = all_k_curves.mean(dim=(0, 1))
    mean_v = all_v_curves.mean(dim=(0, 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))

    axes[0].plot(rank_axis, mean_k, label="K", color="#283593", lw=2)
    axes[0].plot(rank_axis, mean_v, label="V", color="#B23A00", lw=2)
    axes[0].axhline(0.95, color="gray", ls=":", label="95% energy")
    axes[0].set(
        xlabel="matrix rank r",
        ylabel="cumulative squared singular-value energy",
        title="Mean spectrum over all layers and KV heads",
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    layers = [row["layer"] for row in rows]
    axes[1].plot(
        layers, [row["K95_mean"] for row in rows], "o-", label="K", color="#283593"
    )
    axes[1].plot(
        layers, [row["V95_mean"] for row in rows], "s--", label="V", color="#B23A00"
    )
    axes[1].axhline(all_k_curves.shape[-1], color="gray", ls=":", label="full rank")
    axes[1].set(
        xlabel="decoder layer",
        ylabel="mean rank for 95% energy",
        title="Effective rank changes with depth",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    k_sv = torch.linalg.svdvals(selected_k[head].float()).detach().cpu()
    v_sv = torch.linalg.svdvals(selected_v[head].float()).detach().cpu()
    axes[2].semilogy(
        rank_axis, k_sv / k_sv[0], "o-", ms=3, label="K", color="#283593"
    )
    axes[2].semilogy(
        rank_axis, v_sv / v_sv[0], "s--", ms=3, label="V", color="#B23A00"
    )
    axes[2].set(
        xlabel="singular-value index",
        ylabel="singular value / largest singular value",
        title=f"Layer {layer}, KV head {head}",
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def plot_error_comparison(
    comparisons: dict[str, dict[str, list[dict[str, Any]]]],
    output_path: Path,
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    colors = {"per_head": "#283593", "flat": "#B23A00", "tensor": "#2E7D32"}

    for axis, label in zip(axes, ("K", "V")):
        matrix_rows = comparisons[label]["per_head_svd"]
        flat_rows = comparisons[label]["flattened_svd"]
        tucker_rows = comparisons[label]["tucker"]
        axis.plot(
            [row["compression"] for row in matrix_rows],
            [row["relative_error"] for row in matrix_rows],
            "o-",
            color=colors["per_head"],
            label="per-head matrix SVD",
        )
        axis.plot(
            [row["compression"] for row in flat_rows],
            [row["relative_error"] for row in flat_rows],
            "D-.",
            color=colors["flat"],
            label="flattened [token, head×feature] SVD",
        )
        axis.plot(
            [row["compression"] for row in tucker_rows],
            [row["relative_error"] for row in tucker_rows],
            "s--",
            color=colors["tensor"],
            label="joint Tucker/HOSVD",
        )
        for row in matrix_rows:
            axis.annotate(
                f"r={row['rank']}",
                (row["compression"], row["relative_error"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set(
            xlabel="compression ratio (dense scalars / stored scalars)",
            title=f"{label}: matched-storage comparison",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        axis.margins(x=0.08, y=0.10)
    axes[0].set_ylabel("relative Frobenius reconstruction error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def plot_reconstruction_heatmaps(
    original: dict[str, torch.Tensor],
    svd_reconstructed: dict[str, torch.Tensor],
    tucker_reconstructed: dict[str, torch.Tensor],
    layer: int,
    head: int,
    compare_rank: int,
    tucker_ranks: dict[str, tuple[int, int, int]],
    output_path: Path,
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    for row_index, label in enumerate(("K", "V")):
        tensors = (
            original[label][head],
            svd_reconstructed[label][head],
            tucker_reconstructed[label][head],
        )
        max_abs = float(original[label][head].abs().max().item())
        titles = (
            f"Original {label}",
            f"Per-head SVD, r={compare_rank}",
            f"Tucker ranks={tucker_ranks[label]}",
        )
        for column, (tensor, title) in enumerate(zip(tensors, titles)):
            image = axes[row_index, column].imshow(
                tensor.detach().float().cpu().T,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xlabel("token position")
            axes[row_index, column].set_ylabel("head feature")
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.04)
    fig.suptitle(f"Layer {layer}, KV head {head}: same storage budget within each row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def main() -> None:
    args = parse_args()
    if args.tokens < 2:
        raise ValueError("--tokens must be at least 2.")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    device = choose_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    prompt = load_prompt(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    section("1. Load Llama-3.2 and inspect its attention anatomy")
    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")

    # Transformers 4.57 may still request Hub metadata while loading a tokenizer by
    # repository ID, even with local_files_only=True. Resolving the cached snapshot to a
    # real directory first makes the offline behavior unambiguous.
    model_source = args.model
    if args.local_files_only:
        model_source = snapshot_download(args.model, local_files_only=True)
        print(f"Cache:  {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    config = model.config
    num_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    queries_per_kv = query_heads // kv_heads

    first_attention = model.model.layers[0].self_attn
    print(
        "\nCompact architecture:\n"
        f"  tokens -> embedding width {hidden_size}\n"
        f"  -> {num_layers} decoder blocks\n"
        "       each block: self-attention + MLP with residual connections\n"
        "  -> final normalization -> vocabulary logits\n"
    )
    print(
        "Inside each self-attention layer:\n"
        f"  query heads = {query_heads}, KV heads = {kv_heads}, head dimension = {head_dim}\n"
        f"  Grouped-Query Attention: {queries_per_kv} query heads share each K/V head\n"
        f"  W_Q weight shape = {tuple(first_attention.q_proj.weight.shape)}\n"
        f"  W_K weight shape = {tuple(first_attention.k_proj.weight.shape)}\n"
        f"  W_V weight shape = {tuple(first_attention.v_proj.weight.shape)}\n"
        f"  W_O weight shape = {tuple(first_attention.o_proj.weight.shape)}"
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.tokens,
        add_special_tokens=True,
    )
    actual_tokens = int(encoded.input_ids.shape[1])
    inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    print(f"\nPrompt tokens requested/available: {args.tokens}/{actual_tokens}")
    if actual_tokens < 64:
        print(
            "WARNING: fewer than 64 tokens limits matrix rank by sequence length. "
            "Use a longer prompt for a clearer demonstration."
        )

    with torch.inference_mode():
        output = model(**inputs, use_cache=True)
    cache = output.past_key_values

    key0, value0 = cache[0]
    batch, observed_kv_heads, sequence, observed_head_dim = key0.shape
    assert observed_kv_heads == kv_heads and observed_head_dim == head_dim
    bytes_per_element = key0.element_size()
    bytes_per_token = 2 * num_layers * kv_heads * head_dim * bytes_per_element
    total_cache_bytes = bytes_per_token * sequence * batch

    print(
        "\nActual cache returned by Transformers:\n"
        f"  number of layer caches: {len(cache)}\n"
        f"  K shape in every layer: [batch, KV heads, tokens, head feature]\n"
        f"                        = {tuple(key0.shape)}\n"
        f"  V shape in every layer: {tuple(value0.shape)}\n"
        f"  cached K is post-RoPE in this Llama implementation; V is not rotated\n"
        f"  KV bytes per token across all layers: {human_bytes(bytes_per_token)}\n"
        f"  KV bytes for this prompt: {human_bytes(total_cache_bytes)}"
    )

    section("2. Does the real cache look low rank?")
    print(
        "For each layer and KV head, regard K or V as a matrix [tokens, head_feature].\n"
        "The rank@95% is the smallest number of singular directions retaining 95% of\n"
        "the squared singular-value energy. Full possible rank here is "
        f"min(tokens={sequence}, head_dim={head_dim}) = {min(sequence, head_dim)}.\n"
    )
    rank_rows, all_k_curves, all_v_curves = analyze_all_layers(cache)
    print_layer_rank_table(rank_rows)

    selected_layer = num_layers // 2 if args.layer == -1 else args.layer
    if not 0 <= selected_layer < num_layers:
        raise ValueError(f"--layer must be -1 or between 0 and {num_layers - 1}.")
    if not 0 <= args.head < kv_heads:
        raise ValueError(f"--head must be between 0 and {kv_heads - 1}.")

    selected_k = cache[selected_layer][0][0].detach()
    selected_v = cache[selected_layer][1][0].detach()
    maximum_rank = min(sequence, head_dim)
    ranks = parse_ranks(args.ranks, maximum_rank)
    compare_rank = min(args.compare_rank, maximum_rank)
    if compare_rank not in ranks:
        ranks = sorted(set(ranks + [compare_rank]))

    section("3. Matrix SVD versus tensor Tucker/HOSVD")
    print(
        f"Detailed layer: {selected_layer}; tensor shape [H,S,D] = "
        f"{tuple(selected_k.shape)}\n\n"
        "Matrix SVD:\n"
        "  - separately factorizes each KV head's [S,D] matrix;\n"
        "  - does not share a basis across heads;\n"
        "  - stored scalars = H * r * (S + D).\n\n"
        "Flattened matrix SVD:\n"
        "  - reshapes the tensor to [S,H*D];\n"
        "  - shares one token basis across all heads/features;\n"
        "  - stored scalars = r_flat * (S + H*D), but head/feature modes are merged.\n\n"
        "Tensor Tucker/HOSVD:\n"
        "  - jointly factorizes one [H,S,D] tensor;\n"
        "  - can share structure across heads, token positions, and features;\n"
        "  - stores a core [r_H,r_S,r_D] plus three factor matrices;\n"
        "  - stored scalars = r_H*r_S*r_D + H*r_H + S*r_S + D*r_D.\n\n"
        "Because SVD rank and Tucker's three ranks are not comparable names, each Tucker\n"
        "point below is automatically chosen under the SAME scalar budget as its SVD point."
    )

    comparisons: dict[str, dict[str, list[dict[str, Any]]]] = {}
    svd_factors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    tucker_factors: dict[
        str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}

    for label, tensor in (("K", selected_k), ("V", selected_v)):
        matrix_rows, factors = per_head_svd_analysis(tensor, ranks)
        flat_rows = flattened_svd_analysis(tensor, matrix_rows)
        bases_and_core = mode_bases_and_core(tensor)
        tucker_rows = matched_tucker_points(tensor, matrix_rows, bases_and_core[3])
        comparisons[label] = {
            "per_head_svd": matrix_rows,
            "flattened_svd": flat_rows,
            "tucker": tucker_rows,
        }
        svd_factors[label] = factors
        tucker_factors[label] = bases_and_core
        print_comparison(label, matrix_rows, flat_rows, tucker_rows)

    print(
        "\nHow to read the comparison:\n"
        "  - Lower error at the same compression is better.\n"
        "  - Flattened SVD shows what one shared token basis can do without Tucker.\n"
        "  - If flattened SVD wins, shared token structure explains the gain and Tucker's\n"
        "    extra mode constraints are not helping at that budget.\n"
        "  - If Tucker wins, keeping and jointly allocating separate tensor modes is useful.\n"
        "  - If per-head SVD wins, the heads do not share enough exploitable structure\n"
        "    under this Tucker model to justify its extra coupling.\n"
        "  - Neither result proves downstream generation quality."
    )

    original = {"K": selected_k.float(), "V": selected_v.float()}
    svd_reconstructed: dict[str, torch.Tensor] = {}
    tucker_reconstructed: dict[str, torch.Tensor] = {}
    shown_tucker_ranks: dict[str, tuple[int, int, int]] = {}
    for label in ("K", "V"):
        svd_reconstructed[label] = svd_reconstruct(svd_factors[label], compare_rank)
        tucker_row = next(
            row
            for row in comparisons[label]["tucker"]
            if row["matched_to_svd_rank"] == compare_rank
        )
        chosen_ranks = tuple(int(value) for value in tucker_row["ranks"])
        shown_tucker_ranks[label] = chosen_ranks
        tucker_reconstructed[label] = tucker_reconstruct(
            tucker_factors[label], chosen_ranks
        )
        h_rank, s_rank, d_rank = chosen_ranks
        head_note = (
            "all head directions"
            if h_rank == kv_heads
            else f"{h_rank}/{kv_heads} head directions"
        )
        feature_note = (
            "all feature directions"
            if d_rank == head_dim
            else f"{d_rank}/{head_dim} feature directions"
        )
        print(
            f"\nAt the rank-{compare_rank} SVD budget, {label} Tucker chose "
            f"{chosen_ranks}: {head_note}, token rank {s_rank}/{sequence}, "
            f"and {feature_note}."
        )

    section("4. Save visual explanations")
    overview_path = args.output_dir / "01_kv_low_rank_overview.png"
    comparison_path = args.output_dir / "02_matrix_vs_tensor.png"
    heatmap_path = args.output_dir / "03_reconstruction_heatmaps.png"
    results_path = args.output_dir / "demo_results.json"

    figures = [
        plot_low_rank_overview(
            rank_rows,
            all_k_curves,
            all_v_curves,
            selected_k,
            selected_v,
            selected_layer,
            args.head,
            overview_path,
        ),
        plot_error_comparison(comparisons, comparison_path),
        plot_reconstruction_heatmaps(
            original,
            svd_reconstructed,
            tucker_reconstructed,
            selected_layer,
            args.head,
            compare_rank,
            shown_tucker_ranks,
            heatmap_path,
        ),
    ]

    result_document = {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": actual_tokens,
        "architecture": {
            "layers": num_layers,
            "hidden_size": hidden_size,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "queries_per_kv_head": queries_per_kv,
            "head_dim": head_dim,
            "cache_shape_per_layer": list(key0.shape),
            "kv_bytes_per_token": bytes_per_token,
        },
        "rank_summary": rank_rows,
        "selected_layer": selected_layer,
        "selected_head": args.head,
        "comparisons": comparisons,
        "notes": [
            "Cached keys are post-RoPE in this Transformers Llama implementation.",
            "Compression ratios count stored scalars, not kernel metadata or physical packing.",
            "Reconstruction error is structural evidence, not downstream-quality evidence.",
        ],
    }
    results_path.write_text(json.dumps(result_document, indent=2), encoding="utf-8")

    print(f"Saved: {overview_path.resolve()}")
    print(f"Saved: {comparison_path.resolve()}")
    print(f"Saved: {heatmap_path.resolve()}")
    print(f"Saved: {results_path.resolve()}")

    section("5. Final beginner takeaway")
    mean_k95 = sum(row["K95_mean"] for row in rank_rows) / len(rank_rows)
    mean_v95 = sum(row["V95_mean"] for row in rank_rows) / len(rank_rows)
    print(
        f"1. This model has a maximum per-head matrix rank of {maximum_rank} in this run.\n"
        f"2. K needs about {mean_k95:.1f} directions on average for 95% energy.\n"
        f"3. V needs about {mean_v95:.1f} directions on average for 95% energy.\n"
        "4. Therefore 'the KV cache is low rank' is not a yes/no statement: it depends on\n"
        "   K versus V, layer, head, prompt, target error, and decomposition format.\n"
        "5. Matrix SVD asks whether each head is low rank independently.\n"
        "6. Flattened SVD asks whether all heads can share one token-position basis.\n"
        "7. Tucker asks the mode-aware question: which head, token, and feature directions\n"
        "   can be compressed jointly?\n"
        "8. Always compare at matched stored bytes/scalars and then test real model quality."
    )

    if not args.no_show:
        import matplotlib.pyplot as plt

        try:
            plt.show()
        except Exception as exc:  # Saved files remain usable in headless environments.
            print(f"Interactive display failed ({exc}); the figures were still saved.")
    else:
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
