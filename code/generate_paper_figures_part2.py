"""Generate the reliability and risk-coverage figures from cached data.

File names keep the historical fig3/fig4/fig5 numbering; in the camera-ready
manuscript fig4_risk_coverage.pdf is Figure 3 and fig5_reliability_grid.pdf is
Figure 4 (Appendix G). fig3_reliability_7b_vs_70b.pdf is not included in the
camera-ready paper and is generated only for completeness.

fig3_reliability_7b_vs_70b.pdf: Reliability 2-panel — Qwen-7B NQ + Llama-70B PopQA (not in the paper)
fig4_risk_coverage.pdf (paper Fig. 3): Risk-coverage 2-panel — Llama-70B NQ + PopQA
fig5_reliability_grid.pdf (paper Fig. 4): 3x3 reliability grid — all 9 (LLM, dataset) cells
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = "./figures"
DATA_DIR = "./data"
EVIDENCE_CACHE_DIR = os.path.join(DATA_DIR, "evidence_cache")

DATASETS = {
    "nq": os.path.join(DATA_DIR, "nq_cosine.json"),
    "triviaqa": os.path.join(DATA_DIR, "triviaqa_cosine.json"),
    "popqa": os.path.join(DATA_DIR, "popqa_contriever.json"),
}
DATASET_LABEL = {"nq": "NQ", "triviaqa": "TQA", "popqa": "PopQA"}

LLM_INFO = {
    "qwen2_5_7b_instruct_turbo": {
        "label": "Qwen-7B",
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "cache": "llm_cache_qwen2_5_7b_instruct_turbo.json",
    },
    "gpt_4_1_mini": {
        "label": "GPT-4.1-mini",
        "model": "gpt-4.1-mini",
        "cache": "llm_cache_gpt_4_1_mini.json",
    },
    "llama_3_3_70b_instruct_turbo": {
        "label": "Llama-70B",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "cache": "llm_cache_llama_3_3_70b_instruct_turbo.json",
    },
}

BETA = 0.5
LAMBDA = 30.0


# ============================================================
# Helpers (replicated minimally from the project code)
# ============================================================
import re
import string


def normalize_answer(s):
    s = str(s) if s is not None else ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(pred, gold):
    if pred is None:
        return 0
    pred_n = normalize_answer(pred)
    if isinstance(gold, list):
        return int(any(pred_n == normalize_answer(g) for g in gold))
    return int(pred_n == normalize_answer(gold))


def naive_weights(k):
    return [1.0 / k] * k


def dirichlet_weights(sims, evidence, beta, lam):
    numer = [np.exp(beta * s) + lam * e for s, e in zip(sims, evidence)]
    total = sum(numer)
    if total <= 0:
        return naive_weights(len(sims))
    return [n / total for n in numer]


def aggregate_vote(cached_entries, weights):
    answers = [e["answer_text"] for e in cached_entries]
    total_weight = float(sum(weights))
    answer_scores = {}
    for answer, weight in zip(answers, weights):
        normed = normalize_answer(answer)
        if normed and normed != "unknown":
            answer_scores[normed] = answer_scores.get(normed, 0.0) + weight
    if not answer_scores:
        return "", 0.0
    best_normed = max(answer_scores, key=answer_scores.get)
    confidence = answer_scores[best_normed] / total_weight if total_weight > 0 else 0.0
    for answer in answers:
        if normalize_answer(answer) == best_normed:
            return answer, confidence
    return best_normed, confidence


# ============================================================
# Data loading via direct cache access
# ============================================================
import hashlib


def cache_key(model_name, question, doc_text):
    doc_text = doc_text[:800]
    raw = f"{model_name}|||{question}|||{doc_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_data(dataset_name):
    with open(DATASETS[dataset_name]) as f:
        return json.load(f)


def load_llm_cache(llm_slug):
    with open(os.path.join(DATA_DIR, LLM_INFO[llm_slug]["cache"])) as f:
        return json.load(f)


def load_evidence(dataset_name, method):
    path = os.path.join(EVIDENCE_CACHE_DIR, f"{dataset_name}_{method}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_cached_answers(sample, cache, model_name):
    """Return list of {answer_text, score} per doc using cache."""
    q = sample["question"]
    out = []
    for doc in sample["retrieved_docs"]:
        doc_text = doc.get("text", "")
        key = cache_key(model_name, q, doc_text)
        entry = cache.get(key, {})
        out.append({
            "answer_text": entry.get("answer_text", ""),
            "score": doc.get("score", 0.0),
        })
    return out


def compute_method_predictions(dataset_name, llm_slug, method_name):
    """Return (preds, confs, ems, gts)."""
    data = load_data(dataset_name)
    cache = load_llm_cache(llm_slug)
    model_name = LLM_INFO[llm_slug]["model"]

    evidence = None
    if method_name == "Dir-CE":
        evidence = load_evidence(dataset_name, "cross_encoder")
        if evidence is None:
            raise RuntimeError(f"Missing evidence cache for {dataset_name}")

    preds, confs, ems = [], [], []
    for i, sample in enumerate(data):
        cached = get_cached_answers(sample, cache, model_name)
        k = len(cached)
        if method_name == "Naive":
            w = naive_weights(k)
        elif method_name == "Dir-CE":
            sims = [d["score"] for d in sample["retrieved_docs"]]
            ev_i = evidence[i] if i < len(evidence) else [0.0] * k
            w = dirichlet_weights(sims, ev_i, beta=BETA, lam=LAMBDA)
        else:
            raise ValueError(method_name)
        ans, conf = aggregate_vote(cached, w)
        preds.append(ans)
        confs.append(conf)
        gt = sample["answers"]
        ems.append(exact_match(ans, gt))

    return np.array(preds), np.array(confs, dtype=float), np.array(ems, dtype=float)


# ============================================================
# Plot helpers
# ============================================================
def compute_reliability_bins(confs, ems, n_bins=10):
    # Bin assignment via integer floor, matching metrics.expected_calibration_error.
    # (Avoids floating-point boundary drift between linspace edges and confidence
    # values that land exactly on .1, .2, ..., .9.)
    confs = np.clip(confs, 0.0, 1.0)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_accs.append(ems[mask].mean())
            bin_confs.append(confs[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            bin_counts.append(0)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    return bin_centers, bin_accs, bin_confs, bin_counts


def compute_ece(confs, ems, n_bins=10):
    # Bin assignment via integer floor, matching metrics.expected_calibration_error.
    confs = np.clip(confs, 0.0, 1.0)
    n = len(confs)
    bin_ids = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = ems[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def plot_reliability_ax(ax, confs_n, ems_n, confs_d, ems_d, title,
                          show_legend=False, ece_box=True):
    """Plot Naive (gray) vs Dir-CE (green) reliability on given axis."""
    bc_n, ba_n, _, _ = compute_reliability_bins(confs_n, ems_n)
    bc_d, ba_d, _, _ = compute_reliability_bins(confs_d, ems_d)
    ece_n = compute_ece(confs_n, ems_n)
    ece_d = compute_ece(confs_d, ems_d)

    width = 0.04
    ax.bar(np.array(bc_n) - width/2, ba_n, width=width, alpha=0.8,
           color="#7f7f7f", label="Naive", edgecolor="black", linewidth=0.3)
    ax.bar(np.array(bc_d) + width/2, ba_d, width=width, alpha=0.8,
           color="#2ca02c", label="Dir-CE", edgecolor="black", linewidth=0.3)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    if ece_box:
        arrow = "↑" if ece_d > ece_n else "↓"
        ece_text = f"ECE: {ece_n:.3f} $\\to$ {ece_d:.3f} {arrow}"
        ax.text(0.97, 0.03, ece_text, transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="gray", alpha=0.9))
    if show_legend:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95, edgecolor="gray")


def plot_risk_coverage_ax(ax, confs_n, ems_n, confs_d, ems_d, title,
                            aurc_n_override=None, aurc_d_override=None,
                            show_legend=False):
    """Plot risk-coverage curves on given axis. Supports AURC override for
    consistency with Table 5 values (computed via metrics.compute_aurc in the
    main result pipeline)."""
    def curve(confs, ems):
        n = len(confs)
        sorted_idx = np.argsort(-confs)
        sorted_correct = ems[sorted_idx]
        coverages = np.linspace(1 / n, 1.0, n)
        risks = np.array([1.0 - np.mean(sorted_correct[:i + 1]) for i in range(n)])
        return coverages, risks

    cov_n, r_n = curve(confs_n, ems_n)
    cov_d, r_d = curve(confs_d, ems_d)
    aurc_n = aurc_n_override if aurc_n_override is not None else float(np.trapezoid(r_n, cov_n))
    aurc_d = aurc_d_override if aurc_d_override is not None else float(np.trapezoid(r_d, cov_d))

    ax.plot(cov_n, r_n, color="#7f7f7f", linestyle="--", linewidth=1.5,
            label=f"Naive (AURC={aurc_n:.3f})")
    ax.plot(cov_d, r_d, color="#2ca02c", linestyle="-", linewidth=1.5,
            label=f"Dir-CE (AURC={aurc_d:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage", fontsize=9)
    ax.set_ylabel("Risk (1$-$Accuracy)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    if show_legend:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95, edgecolor="gray")


# ============================================================
# fig3_reliability_7b_vs_70b.pdf: Reliability 2-panel (not used in the camera-ready paper)
# ============================================================
def make_fig3():
    print("\n=== Fig 3: Reliability 7B vs 70B ===")
    cells = [
        ("qwen2_5_7b_instruct_turbo", "nq",
         "Qwen-7B, NQ (calibration worsens)"),
        ("llama_3_3_70b_instruct_turbo", "popqa",
         "Llama-70B, PopQA (calibration improves)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (llm, ds, title) in zip(axes, cells):
        print(f"  Computing {llm} on {ds}...")
        _, confs_n, ems_n = compute_method_predictions(ds, llm, "Naive")
        _, confs_d, ems_d = compute_method_predictions(ds, llm, "Dir-CE")
        plot_reliability_ax(ax, confs_n, ems_n, confs_d, ems_d,
                             title, show_legend=True)
        ece_n = compute_ece(confs_n, ems_n)
        ece_d = compute_ece(confs_d, ems_d)
        print(f"    ECE: Naive={ece_n:.4f}, Dir-CE={ece_d:.4f}, ΔECE={ece_d-ece_n:+.4f}")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_reliability_7b_vs_70b.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# fig4_risk_coverage.pdf: Risk-Coverage 2-panel (paper Figure 3, Appendix G)
# ============================================================
def make_fig4():
    """Risk-coverage for Llama-70B × {NQ, PopQA}. Override AURC with values
    from the main voting result JSONs to ensure consistency with Table 5."""
    print("\n=== Fig 4: Risk-Coverage curves (appendix) ===")
    # Source-of-truth AURC values (from run_phase2_analysis using metrics.compute_aurc)
    aurc_overrides = {}
    for llm, ds in [("llama_3_3_70b_instruct_turbo", "nq"),
                     ("llama_3_3_70b_instruct_turbo", "popqa")]:
        path = f"./results/{ds}_cross_encoder_voting_{llm}.json"
        with open(path) as f:
            r = json.load(f)
        aurc_overrides[(llm, ds)] = (
            r["naive"]["AURC"],
            r["dirichlet_b0.5_l30.0_cross_encoder"]["AURC"],
        )

    cells = [
        ("llama_3_3_70b_instruct_turbo", "nq", "Llama-70B, NQ"),
        ("llama_3_3_70b_instruct_turbo", "popqa", "Llama-70B, PopQA"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (llm, ds, title) in zip(axes, cells):
        print(f"  Computing {llm} on {ds}...")
        _, confs_n, ems_n = compute_method_predictions(ds, llm, "Naive")
        _, confs_d, ems_d = compute_method_predictions(ds, llm, "Dir-CE")
        aurc_n, aurc_d = aurc_overrides[(llm, ds)]
        print(f"    AURC (source): Naive={aurc_n:.4f}, Dir-CE={aurc_d:.4f}")
        plot_risk_coverage_ax(ax, confs_n, ems_n, confs_d, ems_d,
                               title,
                               aurc_n_override=aurc_n,
                               aurc_d_override=aurc_d,
                               show_legend=True)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_risk_coverage.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# fig5_reliability_grid.pdf: 3x3 Reliability Grid (paper Figure 4, Appendix G)
# ============================================================
def make_fig5():
    print("\n=== Fig 5: 3x3 Reliability Grid (appendix) ===")
    llms = [
        ("qwen2_5_7b_instruct_turbo", "Qwen-7B"),
        ("gpt_4_1_mini", "GPT-4.1-mini"),
        ("llama_3_3_70b_instruct_turbo", "Llama-70B"),
    ]
    dss = [("nq", "NQ"), ("triviaqa", "TQA"), ("popqa", "PopQA")]

    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.0))
    for r, (llm, llm_label) in enumerate(llms):
        for c, (ds, ds_label) in enumerate(dss):
            print(f"  Computing {llm_label} on {ds_label}...")
            _, confs_n, ems_n = compute_method_predictions(ds, llm, "Naive")
            _, confs_d, ems_d = compute_method_predictions(ds, llm, "Dir-CE")
            plot_reliability_ax(axes[r, c], confs_n, ems_n, confs_d, ems_d,
                                 f"{llm_label}, {ds_label}",
                                 show_legend=(r == 0 and c == 0))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_reliability_grid.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    make_fig3()
    make_fig4()
    make_fig5()
    print("\nDone!")
