"""Generate all paper figures (1, 2) from existing JSON result files.

Fig 1: Signal Comparison Heatmap (3 signals x 9 cells)
Fig 2: Lambda sensitivity curves for CE/ES/NLI (averaged across cells)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

DATASETS = ["nq", "triviaqa", "popqa"]
DATASET_LABEL = {"nq": "NQ", "triviaqa": "TQA", "popqa": "PopQA"}

LLMS = [
    ("qwen2_5_7b_instruct_turbo", "Qwen-7B"),
    ("gpt_4_1_mini", "GPT-4.1m"),
    ("llama_3_3_70b_instruct_turbo", "Llama-70B"),
]

SIGNALS = [
    ("cross_encoder", "CE"),
    ("embedding_stability", "ES"),
    ("nli", "NLI"),
]

LAMBDAS = [0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]  # index 0 means REPLUG (no lambda)
BETA_DEFAULT = "0.5"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_ce_result_file(ds, llm_slug, signal_slug):
    return f"{RESULTS_DIR}/{ds}_{signal_slug}_voting_{llm_slug}.json"


def extract_em(data, key):
    if key not in data:
        return None
    return data[key].get("EM")


# ============================================================
# Build master matrix: delta_em[signal, dataset, llm]
# ============================================================
def build_dem_matrix():
    """Return dict: dem_matrix[signal][dataset][llm] = ΔEM%p at (β=0.5, λ=30)."""
    naive_em = {}  # naive_em[dataset][llm]
    dir_em = {}    # dir_em[signal][dataset][llm]

    for ds in DATASETS:
        naive_em[ds] = {}
        for llm_slug, llm_label in LLMS:
            # Any signal's file has the naive baseline; use CE as canonical
            path = get_ce_result_file(ds, llm_slug, "cross_encoder")
            data = load_json(path)
            naive_em[ds][llm_label] = extract_em(data, "naive") * 100  # as percent

    for sig_slug, sig_label in SIGNALS:
        dir_em[sig_label] = {}
        for ds in DATASETS:
            dir_em[sig_label][ds] = {}
            for llm_slug, llm_label in LLMS:
                path = get_ce_result_file(ds, llm_slug, sig_slug)
                data = load_json(path)
                # Key: dirichlet_b0.5_l30.0_{signal}
                key = f"dirichlet_b{BETA_DEFAULT}_l30.0_{sig_slug}"
                em = extract_em(data, key)
                dir_em[sig_label][ds][llm_label] = em * 100 if em is not None else np.nan

    # Compute ΔEM
    dem = {}
    for sig_label in [s[1] for s in SIGNALS]:
        dem[sig_label] = {}
        for ds in DATASETS:
            dem[sig_label][ds] = {}
            for _, llm_label in LLMS:
                dem[sig_label][ds][llm_label] = (
                    dir_em[sig_label][ds][llm_label] - naive_em[ds][llm_label]
                )
    return dem, naive_em, dir_em


# ============================================================
# Fig 1: Signal Comparison Heatmap
# ============================================================
def plot_signal_heatmap(dem, save_path):
    """3 signals (rows) x 9 cells (cols) heatmap of ΔEM."""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))

    # Build matrix
    sig_labels = ["CE", "ES", "NLI"]
    cell_labels = []
    matrix = np.zeros((3, 9))
    for j_ds, ds in enumerate(DATASETS):
        for j_llm, (_, llm_label) in enumerate(LLMS):
            col_idx = j_ds * 3 + j_llm
            cell_labels.append(f"{llm_label.replace('-', '-\n')}")
            for i, sig in enumerate(sig_labels):
                matrix[i, col_idx] = dem[sig][ds][llm_label]

    # Diverging colormap centered at 0 (colorblind-friendly: RdBu_r)
    vmin, vmax = -3.0, 5.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    # Ticks — rotate x labels to avoid overlap
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels([f"{l}" for _, l in LLMS] * 3, fontsize=8,
                        rotation=35, ha="right")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(sig_labels, fontsize=10, fontweight="bold")

    # Dataset group labels above (positioned higher to avoid title overlap)
    for j_ds, ds in enumerate(DATASETS):
        x_center = j_ds * 3 + 1
        ax.text(x_center, -1.05, DATASET_LABEL[ds], ha="center", va="center",
                fontsize=11, fontweight="bold")
    # Vertical separators between datasets
    for j in [3, 6]:
        ax.axvline(j - 0.5, color="black", linewidth=1.2)

    # Cell values
    for i in range(3):
        for j in range(9):
            v = matrix[i, j]
            color = "white" if abs(v) > 2.5 else "black"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\Delta$EM (\%p vs Naive)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(r"Signal comparison: $\Delta$EM across 9 (LLM, dataset) cells",
                 fontsize=11, pad=28)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    print("Matrix values:")
    for i, sig in enumerate(sig_labels):
        row_vals = ", ".join(f"{matrix[i, j]:+.2f}" for j in range(9))
        print(f"  {sig}: {row_vals}")


# ============================================================
# Fig 2: Lambda sensitivity for 3 signals (averaged)
# ============================================================
def build_lambda_sensitivity():
    """For each signal, compute average ΔEM across 9 cells at each λ."""
    signal_curves = {}  # signal_curves[signal_label] = list of (lambda, mean_dEM, std_dEM)
    for sig_slug, sig_label in SIGNALS:
        curves = []
        for lam in LAMBDAS:
            dems = []
            for ds in DATASETS:
                for llm_slug, llm_label in LLMS:
                    path = get_ce_result_file(ds, llm_slug, sig_slug)
                    data = load_json(path)
                    naive = data["naive"]["EM"] * 100
                    if lam == 0:
                        # REPLUG at beta=0.5
                        key = f"replug_b{BETA_DEFAULT}"
                    else:
                        lam_str = f"{lam:.1f}"
                        key = f"dirichlet_b{BETA_DEFAULT}_l{lam_str}_{sig_slug}"
                    em = extract_em(data, key)
                    if em is None:
                        continue
                    dems.append(em * 100 - naive)
            if dems:
                curves.append((lam, np.mean(dems), np.std(dems) / np.sqrt(len(dems))))
        signal_curves[sig_label] = curves
    return signal_curves


def plot_lambda_sensitivity(curves, save_path):
    """3 curves (CE/ES/NLI) showing ΔEM vs λ averaged across 9 cells."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    colors = {"CE": "#2ca02c", "ES": "#1f77b4", "NLI": "#d62728"}
    markers = {"CE": "o", "ES": "s", "NLI": "^"}
    for sig, curve in curves.items():
        lams = [c[0] for c in curve]
        means = [c[1] for c in curve]
        sems = [c[2] for c in curve]
        # Replace lambda=0 with small value for log-scale plotting
        lams_plot = [0.05 if l == 0 else l for l in lams]
        ax.errorbar(lams_plot, means, yerr=sems, color=colors[sig],
                    marker=markers[sig], markersize=5, linewidth=1.5,
                    capsize=2, label=sig)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ (evidence weight)", fontsize=9)
    ax.set_ylabel(r"$\Delta$EM (\%p vs Naive)", fontsize=9)
    ax.set_xticks([0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0])
    ax.set_xticklabels(["0", "0.1", "0.5", "1", "3", "10", "30"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95, edgecolor="gray")
    ax.set_title(r"$\lambda$ sensitivity ($\beta$=0.5, avg over 9 cells)",
                 fontsize=9.5, pad=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    print("Curve values:")
    for sig, curve in curves.items():
        for lam, mean, sem in curve:
            print(f"  {sig} λ={lam}: mean={mean:+.3f}, sem={sem:.3f}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Building ΔEM matrix...")
    dem, naive_em, dir_em = build_dem_matrix()

    print("\nVerification — Naive EMs:")
    for ds in DATASETS:
        for _, llm in LLMS:
            print(f"  {DATASET_LABEL[ds]:6s} {llm:15s} Naive EM = {naive_em[ds][llm]:.2f}")

    print("\nVerification — Dir-CE EMs:")
    for ds in DATASETS:
        for _, llm in LLMS:
            print(f"  {DATASET_LABEL[ds]:6s} {llm:15s} Dir-CE EM = {dir_em['CE'][ds][llm]:.2f}")

    print("\n" + "=" * 60)
    print("Plotting Fig 1: Signal Comparison Heatmap")
    plot_signal_heatmap(dem, os.path.join(FIGURES_DIR, "fig1_signal_heatmap.pdf"))

    print("\n" + "=" * 60)
    print("Building λ sensitivity curves...")
    curves = build_lambda_sensitivity()

    print("\n" + "=" * 60)
    print("Plotting Fig 2: Lambda sensitivity 3-signal")
    plot_lambda_sensitivity(curves, os.path.join(FIGURES_DIR, "fig2_lambda_sensitivity.pdf"))

    print("\nDone!")
