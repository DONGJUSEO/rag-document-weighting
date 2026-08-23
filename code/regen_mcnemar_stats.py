"""Regenerate the derived statistics in results/mcnemar_results.json.

The paired counts n01/n10 are the primary data (recomputed from caches by
verify_unchecked_numbers.py, CLAIM 3) and are never modified here. This
script deterministically re-derives, per cell:

  chi2_yates    = (|n01 - n10| - 1)^2 / (n01 + n10)   [Yates continuity correction]
  p_value_yates = chi2.sf(chi2_yates, df=1)           [canonical; matches the paper]
  p_value       = chi2.sf(chi2_uncorrected, df=1)     [reference only]
  significant_Bonf_27 : p_value_yates < alpha/27

chi2.sf is used instead of 1 - chi2.cdf to avoid float underflow to 0.0
for very small p-values (e.g., the PopQA cells).

Run from the package root:  python code/regen_mcnemar_stats.py
"""
import json
import os

from scipy.stats import chi2

PATH = os.path.join(os.path.dirname(__file__), "..", "results", "mcnemar_results.json")
ALPHA_BONF = 0.05 / 27

with open(PATH) as f:
    data = json.load(f)

changed = []
for key, cell in data.items():
    if key == "_meta" or not isinstance(cell, dict):
        continue
    n01, n10 = cell["n01"], cell["n10"]
    n = n01 + n10
    chi2_yates = max(0.0, abs(n01 - n10) - 1) ** 2 / n if n else 0.0  # Yates; clamp matters only when n01 == n10
    chi2_unc = (n01 - n10) ** 2 / n if n else 0.0
    new = {
        "n01": n01,
        "n10": n10,
        "p_value": float(chi2.sf(chi2_unc, df=1)),
        "significant_Bonf_27": bool(chi2.sf(chi2_yates, df=1) < ALPHA_BONF),
        "chi2_yates": float(chi2_yates),
        "p_value_yates": float(chi2.sf(chi2_yates, df=1)),
    }
    # Gate: primary data and significance verdicts must be unchanged.
    assert (new["n01"], new["n10"]) == (n01, n10)
    assert new["significant_Bonf_27"] == cell["significant_Bonf_27"], key
    if any(abs(new[k] - cell[k]) > 0 for k in ("p_value", "p_value_yates", "chi2_yates")):
        changed.append(key)
    data[key] = new

data["_meta"] = {
    "bonferroni_correction": "alpha / 27 (9 LLM×dataset cells × 3 evidence types)",
    "alpha_corrected": ALPHA_BONF,
    "test": "McNemar with Yates continuity correction, chi-squared approximation",
    "formula": "max(0, |n01 - n10| - 1)**2 / (n01 + n10)",
    "fields": "p_value_yates = Yates-corrected p via chi2.sf (canonical; matches the paper); "
              "p_value = uncorrected chi-square p via chi2.sf (reference only); "
              "significant_Bonf_27 uses p_value_yates",
    "regenerated": "2026-07-24 via code/regen_mcnemar_stats.py",
    "n01_definition": "Naive wrong, Dir-CE correct (paired)",
    "n10_definition": "Naive correct, Dir-CE wrong (paired)",
}

with open(PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"updated p/chi2 fields in {len(changed)} cells: {changed}")
