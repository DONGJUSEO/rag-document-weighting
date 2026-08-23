"""
Regenerate the rows of Table 2 (main results) from the result JSONs.

Point estimates (EM, F1, ΔEM) are read from the full-test per-query voting JSONs
written by run_main.py (results/{dataset}_cross_encoder_voting_{llm}.json; see README §1,
regenerated from the released caches). The ±std values are the 3-seed 50/50 cal/test
repeated-split standard deviations (results/naive_simw_splits_{llm}.json for Naive/SimW and
results/phase2_analysis_{llm}.json for Dir-CE). The split mean/std pairs are also written to
results/table1_mean_std.json; the split MEANS are not the point estimates shown in Table 2.

If a voting JSON is missing, the script still writes the split statistics and prints
"n/a" for the affected point estimates.
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
DIR_CE_KEY = "dirichlet_b0.5_l30.0_cross_encoder"   # fixed default beta=0.5, lambda=30
SIMW_KEY = "replug_b0.5"                            # REPLUG-style similarity weighting, beta=0.5


def frac(x):
    """Normalise a metric stored either as a fraction or as a percentage to a fraction."""
    return x / 100.0 if x > 1 else x


def cell(point, std, bold=False):
    """Table cell: point estimate (%, one decimal) + split std (%, two decimals)."""
    p = "n/a" if point is None else f"{point * 100:.1f}"
    if bold and point is not None:
        p = f"\\textbf{{{p}}}"
    return f"{p}{{\\scriptsize$\\pm${std * 100:.2f}}}"


def main():
    naive_simw, dir_ce, voting = {}, {}, {}
    for llm_key, _ in LLMS:
        with open(os.path.join(RESULTS_DIR, f"naive_simw_splits_{llm_key}.json")) as f:
            naive_simw[llm_key] = json.load(f)
        # phase2_analysis files use 'llama_3_3_70b' rather than 'llama_3_3_70b_instruct_turbo'
        with open(os.path.join(RESULTS_DIR, f"phase2_analysis_{llm_key.replace('_instruct_turbo', '')}.json")) as f:
            dir_ce[llm_key] = json.load(f)
        for ds_key, _ in DATASETS:
            p = os.path.join(RESULTS_DIR, f"{ds_key}_cross_encoder_voting_{llm_key}.json")
            if os.path.exists(p):
                with open(p) as f:
                    voting[(ds_key, llm_key)] = json.load(f)
            else:
                print(f"[warn] {p} not found; point estimates for {ds_key}/{llm_key} printed as n/a "
                      f"(run code/run_main.py --dataset {ds_key} --evidence cross_encoder first)")

    summary, rows = {}, []
    for ds_key, ds_label in DATASETS:
        for i, (llm_key, llm_label) in enumerate(LLMS):
            ns = naive_simw[llm_key][ds_key]["average"]
            ce = dir_ce[llm_key][f"repeated_split_{ds_key}_cross_encoder"]["average"]
            std = {
                "naive": (ns["naive"]["EM_std"], ns["naive"]["F1_std"]),
                "simw": (ns["simw"]["EM_std"], ns["simw"]["F1_std"]),
                "dir_ce": (frac(ce["EM_std"]), frac(ce["F1_std"])),
            }
            v = voting.get((ds_key, llm_key))
            point = {
                "naive": (v["naive"]["EM"], v["naive"]["F1"]) if v else (None, None),
                "simw": (v[SIMW_KEY]["EM"], v[SIMW_KEY]["F1"]) if v else (None, None),
                "dir_ce": (v[DIR_CE_KEY]["EM"], v[DIR_CE_KEY]["F1"]) if v else (None, None),
            }
            best = {}
            for m in (0, 1):
                vals = [point[k][m] for k in ("naive", "simw", "dir_ce")]
                best[m] = None if any(x is None for x in vals) else max(vals)
            cells = []
            for k in ("naive", "simw", "dir_ce"):
                for m in (0, 1):
                    cells.append(cell(point[k][m], std[k][m],
                                      bold=(best[m] is not None and point[k][m] == best[m])))
            if point["dir_ce"][0] is None:
                delta = "n/a"
            else:
                d = (point["dir_ce"][0] - point["naive"][0]) * 100   # raw difference, then rounded
                delta = f"{'+' if d >= 0 else ''}{d:.1f}"
            prefix = f"\\multirow{{3}}{{*}}{{{ds_label}}} & {llm_label}" if i == 0 else f"& {llm_label}"
            rows.append(f"{prefix} & " + " & ".join(cells) + f" & {delta} \\\\")

            summary[f"{ds_key}_{llm_key}"] = {
                "naive_em_mean": ns["naive"]["EM_mean"] * 100, "naive_em_std": ns["naive"]["EM_std"] * 100,
                "naive_f1_mean": ns["naive"]["F1_mean"] * 100, "naive_f1_std": ns["naive"]["F1_std"] * 100,
                "simw_em_mean": ns["simw"]["EM_mean"] * 100, "simw_em_std": ns["simw"]["EM_std"] * 100,
                "simw_f1_mean": ns["simw"]["F1_mean"] * 100, "simw_f1_std": ns["simw"]["F1_std"] * 100,
                "dir_ce_em_mean": frac(ce["EM_mean"]) * 100, "dir_ce_em_std": frac(ce["EM_std"]) * 100,
                "dir_ce_f1_mean": frac(ce["F1_mean"]) * 100, "dir_ce_f1_std": frac(ce["F1_std"]) * 100,
                "delta_em_mean": (frac(ce["EM_mean"]) - ns["naive"]["EM_mean"]) * 100,
            }
        rows.append(r"\midrule")
    rows.pop()  # trailing midrule

    print("\n% === Table 2 rows (point estimates: full test set; ±std: 3-seed repeated split) ===\n")
    for r in rows:
        print(r)

    out_json = os.path.join(RESULTS_DIR, "table1_mean_std.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSplit statistics (mean/std of the 3-seed splits, NOT the Table 2 point estimates): {out_json}")


if __name__ == "__main__":
    main()
