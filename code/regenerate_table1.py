"""
Regenerate Table 2 (main results) with 3-seed mean ± std for Naive / SimW / Dir-CE.
Produces LaTeX tabular rows + a summary JSON for inline text verification.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

LLMS = [
    ("qwen2_5_7b_instruct_turbo", "Qwen-7B"),
    ("gpt_4_1_mini", "GPT-4.1-mini"),
    ("llama_3_3_70b_instruct_turbo", "Llama-70B"),
]
DATASETS = [("nq", "NQ"), ("triviaqa", "TQA"), ("popqa", "PopQA")]


def fmt_pct(mean_frac, std_frac):
    """mean/std are in [0,1]. Output as percentages with ±."""
    m = mean_frac * 100.0
    s = std_frac * 100.0
    return f"{m:.1f} $\\pm$ {s:.2f}"


def main():
    # Load Naive/SimW from our new files, Dir-CE from existing phase2 files
    naive_simw = {}
    for llm_key, _ in LLMS:
        with open(os.path.join(RESULTS_DIR, f"naive_simw_splits_{llm_key}.json")) as f:
            naive_simw[llm_key] = json.load(f)

    dir_ce = {}
    for llm_key, _ in LLMS:
        # phase2_analysis file uses 'llama_3_3_70b' not 'llama_3_3_70b_instruct_turbo'
        phase_key = llm_key.replace("_instruct_turbo", "")
        with open(os.path.join(RESULTS_DIR, f"phase2_analysis_{phase_key}.json")) as f:
            dir_ce[llm_key] = json.load(f)

    summary = {}
    latex_rows = []
    for ds_key, ds_label in DATASETS:
        for i, (llm_key, llm_label) in enumerate(LLMS):
            ns_avg = naive_simw[llm_key][ds_key]["average"]
            ce_avg = dir_ce[llm_key][f"repeated_split_{ds_key}_cross_encoder"]["average"]

            naive_em = (ns_avg["naive"]["EM_mean"], ns_avg["naive"]["EM_std"])
            naive_f1 = (ns_avg["naive"]["F1_mean"], ns_avg["naive"]["F1_std"])
            simw_em = (ns_avg["simw"]["EM_mean"], ns_avg["simw"]["EM_std"])
            simw_f1 = (ns_avg["simw"]["F1_mean"], ns_avg["simw"]["F1_std"])
            dir_em = (ce_avg["EM_mean"] / 100 if ce_avg["EM_mean"] > 1 else ce_avg["EM_mean"],
                      ce_avg["EM_std"] / 100 if ce_avg["EM_std"] > 1 else ce_avg["EM_std"])
            dir_f1 = (ce_avg["F1_mean"] / 100 if ce_avg["F1_mean"] > 1 else ce_avg["F1_mean"],
                      ce_avg["F1_std"] / 100 if ce_avg["F1_std"] > 1 else ce_avg["F1_std"])

            # ΔEM (Dir-CE vs Naive), using means
            delta_em = (dir_em[0] - naive_em[0]) * 100
            delta_sign = "+" if delta_em >= 0 else ""

            # LaTeX row
            if i == 0:
                prefix = f"\\multirow{{3}}{{*}}{{{ds_label}}} & {llm_label}"
            else:
                prefix = f"& {llm_label}"
            row = (f"{prefix} & {fmt_pct(*naive_em)} & {fmt_pct(*naive_f1)} "
                   f"& {fmt_pct(*simw_em)} & {fmt_pct(*simw_f1)} "
                   f"& \\textbf{{{dir_em[0]*100:.1f} $\\pm$ {dir_em[1]*100:.2f}}} "
                   f"& \\textbf{{{dir_f1[0]*100:.1f} $\\pm$ {dir_f1[1]*100:.2f}}} "
                   f"& {delta_sign}{delta_em:.1f} \\\\")
            latex_rows.append(row)

            summary[f"{ds_key}_{llm_key}"] = {
                "naive_em_mean": naive_em[0] * 100, "naive_em_std": naive_em[1] * 100,
                "naive_f1_mean": naive_f1[0] * 100, "naive_f1_std": naive_f1[1] * 100,
                "simw_em_mean": simw_em[0] * 100, "simw_em_std": simw_em[1] * 100,
                "simw_f1_mean": simw_f1[0] * 100, "simw_f1_std": simw_f1[1] * 100,
                "dir_ce_em_mean": dir_em[0] * 100, "dir_ce_em_std": dir_em[1] * 100,
                "dir_ce_f1_mean": dir_f1[0] * 100, "dir_ce_f1_std": dir_f1[1] * 100,
                "delta_em_mean": delta_em,
            }
        latex_rows.append(r"\midrule")

    # Remove trailing midrule
    if latex_rows and latex_rows[-1] == r"\midrule":
        latex_rows.pop()

    print("\n% === Table 2 rows ===\n")
    for r in latex_rows:
        print(r)

    out_json = os.path.join(RESULTS_DIR, "table1_mean_std.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON: {out_json}")


if __name__ == "__main__":
    main()
