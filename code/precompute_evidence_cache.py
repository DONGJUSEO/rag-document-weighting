#!/usr/bin/env python3
"""
Precompute the per-query evidence caches consumed by the analysis scripts
(compute_c1..c3, compute_gold_subset_analysis, run_phase2_analysis,
run_additional_baselines, verify_unchecked_numbers, ...).

Writes data/evidence_cache/{dataset}_{method}.json: a list aligned with the
dataset's native query order (NO filtering — queries with <10 retrieved docs
keep their shorter doc lists, matching the bundled caches), each entry the
per-document evidence scores for that query.

Determinism: cross_encoder is deterministic per query. embedding_stability
draws Gaussian perturbations from the torch RNG, which the original run
seeded ONCE (SEED=42) and then consumed across all three datasets in order
(nq -> triviaqa -> popqa) within a single process. This script therefore
re-seeds once per METHOD and iterates datasets in that fixed order, which
reproduces the bundled caches exactly. Consequence: regenerating
embedding_stability for a later dataset alone (e.g. --datasets popqa)
yields statistically equivalent but not byte-identical noise, because the
preceding datasets' RNG draws are skipped.

Usage:
  python code/precompute_evidence_cache.py                    # all 3 x 2, full
  python code/precompute_evidence_cache.py --methods cross_encoder
  python code/precompute_evidence_cache.py --verify 12        # check first 12
      queries per (dataset, method) against the bundled files, write nothing
      (for embedding_stability this exactly checks nq only; see above)
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASETS, DATA_DIR, SEED  # noqa: E402
from evidence_scores import compute_all_evidence  # noqa: E402

METHODS = ["cross_encoder", "embedding_stability"]
TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--verify", type=int, default=None, metavar="N",
                    help="compare the first N queries against the existing "
                         "cache files instead of writing anything")
    a = ap.parse_args()

    cache_dir = os.path.join(DATA_DIR, "evidence_cache")
    os.makedirs(cache_dir, exist_ok=True)
    failures = 0

    for method in a.methods:
        torch.manual_seed(SEED)  # one seed per method, then a continuous pass
        for ds in [d for d in DATASETS if d in a.datasets]:
            with open(DATASETS[ds], "r", encoding="utf-8") as f:
                data = json.load(f)
            subset = data[: a.verify] if a.verify else data
            scores = [
                [float(v) for v in compute_all_evidence(
                    s["question"], s["retrieved_docs"], method=method)]
                for s in tqdm(subset, desc=f"{ds}/{method}")
            ]
            out = os.path.join(cache_dir, f"{ds}_{method}.json")
            if a.verify:
                with open(out, "r", encoding="utf-8") as f:
                    ref = json.load(f)[: a.verify]
                bad = sum(
                    1 for got, want in zip(scores, ref)
                    if len(got) != len(want)
                    or any(abs(g - w) > TOL for g, w in zip(got, want))
                )
                status = "OK" if bad == 0 else f"MISMATCH in {bad}/{len(ref)}"
                print(f"  verify {ds}/{method} (first {a.verify}): {status}")
                failures += bad
            else:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(scores, f)
                print(f"  wrote {out} ({len(scores)} queries)")

    if a.verify and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
