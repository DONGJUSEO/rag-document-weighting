# When Retrieval Similarity Hurts: A Diagnostic Framework for Document Weighting in Black-Box RAG

Code and result files for the paper

> Dongju Seo and Beakcheol Jang. **When Retrieval Similarity Hurts: A Diagnostic Framework for Document Weighting in Black-Box RAG.** *Findings of the Association for Computational Linguistics: EMNLP 2026.*

- Paper: ACL Anthology (to appear, November 2026)
- Cached LLM outputs and evidence caches (≈95 MB compressed / 275 MB extracted): Zenodo archive, DOI to be added here once minted (see *Cached outputs* below). With the caches, every number in the paper is reproducible from this code **without any new API calls**.

We propose a unified two-parameter weighting framework that generalizes REPLUG by adding an evidence term, with retrieval-similarity weighting ($\lambda{=}0$) and evidence-only weighting ($\lambda{\to}\infty$) emerging as limiting cases.

---

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Apache 2.0
├── .env.example                       # Environment-variable template
├── .gitignore                         # Standard ignores (env, caches, large data)
├── code/                              # All experiment scripts (40 .py + 3 .sh)
│   ├── weighting.py                  # Core framework (Eq. 2)
│   ├── metrics.py                    # McNemar, F1, EM, AUROC
│   ├── generation.py                 # Black-box LLM API caller (with answer cache)
│   ├── evidence_scores.py            # CE / ES / NLI evidence
│   ├── precompute_evidence_cache.py  # Rebuild data/evidence_cache/ (CE + ES)
│   ├── run_main.py                   # Main 9-cell pipeline (Table 2; per-query EM vectors)
│   ├── compute_c1_*.py ~ c5_*.py    # Mechanism diagnostics (Appendix O)
│   ├── recompute_popqa_k10.py        # PopQA k=10 strict re-eval
│   ├── run_concat_baseline.py        # Concat-prompt baseline (Appendix P)
│   ├── run_bm25_pipeline.py          # BM25 robustness (Appendix Q)
│   ├── run_bm25_grid_search.py       # BM25 24-point (β,λ) grid
│   ├── compute_gold_subset_analysis.py  # Has-gold / no-gold split (§5.2, App. R; also E5)
│   ├── compute_simw_es_mcnemar.py    # SimW-vs-Naive + Dir-ES-vs-Naive McNemar (App. D)
│   ├── compute_nli_mcnemar.py        # Dir-NLI-vs-Naive McNemar + 216-run grid sweep (App. C, Table 14)
│   ├── regen_mcnemar_stats.py        # Re-derive chi2/p in results/mcnemar_results.json
│   ├── runpod_e5_encode_search.py    # E5 corpus encoding + top-10 retrieval (App. S; GPU host)
│   ├── assemble_e5_and_ce.py         # Assemble E5 retrieval JSONs + CE evidence cache
│   ├── run_e5_llm_pairs.py           # Pairs-parallel per-document LLM answers (E5)
│   ├── run_e5_grid_search.py         # E5 24-point (β,λ) grid from cached answers
│   ├── compute_aggregation_baselines.py # RRF / Borda / CE-Top1 / CE-Rerank comparison (App. T)
│   └── ...
└── results/                           # Summary JSONs for paper tables / appendices
    ├── table1_mean_std.json           # Table 2 ±std source: 3-seed repeated-split
    │                                   #   mean/std (NOT the full-test point estimates;
    │                                   #   point estimates in Table 2 use the full test set)
    ├── mcnemar_results.json           # McNemar χ² + Bonferroni (Dir-CE vs Naive, Table 15;
    │                                   #   derived stats regenerable via
    │                                   #   code/regen_mcnemar_stats.py;
    │                                   #   PopQA rows use the full set n=14,267 —
    │                                   #   strict-k counts in the Table 15 caption)
    ├── simw_vs_naive_mcnemar.json     # SimW vs Naive McNemar (App. D; strict k=10)
    ├── dir_es_vs_naive_mcnemar.json   # Dir-ES vs Naive McNemar (App. D; strict k=10)
    ├── dir_nli_vs_naive_mcnemar.json  # Dir-NLI vs Naive McNemar, all 9 cells + 24-config
    │                                   #   grid sweep (App. C, Table 14; full retrieval set)
    ├── c1_answer_support_mass.json … c5_cost_normalized.json   # Appendix O
    ├── additional_baselines_*.json    # Oracle, Random, Evidence-only, CE-Rerank (full set, §4)
    ├── phase2_analysis_*.json         # Transfer, repeated-split, SmoothECE, Brier
    ├── bm25_grid_search.json + bm25_pipeline_qwen_*.json       # Appendix Q
    ├── naive_simw_splits_*.json       # Table 2 ±std (3-seed splits)
    ├── gold_subset_analysis.json      # Has-gold / no-gold decomposition (Table 3, App. R;
    │                                   #   strict k=10 cache re-aggregation — full-bucket EM
    │                                   #   may differ from Table 2 point estimates by ≤0.07
    │                                   #   on GPT cells; see the App. J nondeterminism note)
    ├── gold_subset_analysis_e5.json   # E5 evaluation, all 9 cells (Table 28, App. S)
    ├── e5_grid_search.json            # E5 (β,λ) grid (Table 29, App. S)
    ├── aggregation_baselines.json     # RRF / Borda / CE-Top1 / CE-Rerank (Table 30, App. T)
    ├── weight_entropy.json            # Dir-CE weight entropy (§6.4, App. O)
    ├── direction_*.json, m1_*.json, m2_*.json, nli_direction_auc.json
    │                                   #   (one-off analyses; the paper values they
    │                                   #   feed are re-derivable via
    │                                   #   verify_unchecked_numbers.py / compute_nli_reverse.py)
    └── smooth_ece_v2.json             # SmoothECE (verified by verify_unchecked_numbers.py)
```

## Cached outputs (`data/`) and full per-query JSONs

Only summary JSONs (the values referenced in the paper) are versioned in this
repository. Three artifact groups are distributed separately because of their
size:

1. **Cached LLM outputs** (`data/llm_cache_*.json`, ≈190 MB across 3 LLMs ×
   31,861 queries × top-$k$ docs on DPR/Contriever, plus ≈105 MB for the E5
   re-run, plus the BM25 and concat-prompt baseline caches) — let every
   pipeline skip API calls.
2. **Evidence caches** (`data/evidence_cache/`, ≈19 MB) — per-query CE/ES
   scores consumed by the analysis scripts. Also regenerable deterministically
   with local models only: `python code/precompute_evidence_cache.py`
   (E5 variants via `assemble_e5_and_ce.py`).
3. **Per-query voting JSONs** (`results/*_voting_*.json`, ≈130 MB across 27
   (LLM × dataset × evidence) cells, including the `_per_question` EM vectors
   used by the McNemar scripts) — regenerated by `run_main.py` from the caches
   above (no API calls; only the local evidence models run).

Groups 1–2 are archived on **Zenodo** as three tarballs
(`llm_caches_dpr_contriever.tar.gz`, `llm_caches_e5.tar.gz`,
`evidence_caches.tar.gz`; sha256 in `SHA256SUMS.txt` and a per-file manifest).
Extract them into `data/` at the package root:

```bash
tar xzf llm_caches_dpr_contriever.tar.gz -C data/
tar xzf llm_caches_e5.tar.gz -C data/
tar xzf evidence_caches.tar.gz -C data/     # creates data/evidence_cache/
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 on macOS. Should also work on Linux.

---

## Data Download

We exclude raw test sets due to size and licensing. Reproduce by downloading:

### Required public datasets

1. **NQ-Open** (CC-BY-SA 3.0):
   - Download from https://github.com/google-research-datasets/natural-questions
   - We use 8,757 dev queries.

2. **TriviaQA** (Apache 2.0):
   - Download from https://nlp.cs.washington.edu/triviaqa/
   - We use 8,837 dev queries (rc-wikipedia subset).

3. **PopQA** (MIT):
   - Download from https://huggingface.co/datasets/akariasai/PopQA
   - We use 14,267 queries.

### Retrieved documents

For DPR (NQ/TriviaQA) and Contriever (PopQA) retrieved docs:
- DPR: https://github.com/facebookresearch/DPR (psgs_w100.tsv)
- Contriever: https://github.com/facebookresearch/contriever

The BM25 scripts (Appendix Q) read `data/nq_test.json` / `data/triviaqa_test.json`:
these are the same DPR test-split files as `data/nq_cosine.json` /
`data/triviaqa_cosine.json` (identical schema; only `question`/`answers` are used
for BM25 — copy or symlink the cosine files under the `*_test.json` names).

### BM25 index (Appendix Q)

```bash
# Build Pyserini BM25 index from psgs_w100.tsv (k1=0.9, b=0.4)
python code/build_bm25_pyserini.py
```

### E5 retrieval (Appendix S)

E5 (`intfloat/e5-base-v2`) re-encodes the same `psgs_w100.tsv` corpus
(21M passages, `"query: "` / `"passage: "` prefixes) on a GPU host:

```bash
python code/runpod_e5_encode_search.py prep             # local: export query lists
python code/runpod_e5_encode_search.py run --model e5   # GPU host; writes *_e5_search.json
python code/assemble_e5_and_ce.py --model e5            # local; builds data JSONs + CE evidence
```

Place datasets under `data/` matching the file paths in `code/config.py`.

---

## Reproducing Main Results

Run all commands from the package root (paths inside the scripts resolve
relative to their own location, but a few figure/verification scripts also
use `./data`-style paths that assume the package root as cwd).

### 1. Main results (Table 2, 9 cells × 3 LLMs)

The LLM is selected via the `LLM_MODEL` environment variable; convenience shell scripts are provided for each LLM.

```bash
export TOGETHER_API_KEY=<your_key>      # Qwen, Llama (Together AI) — not needed with the caches
export OPENAI_API_KEY=<your_key>         # GPT-4.1-mini               — not needed with the caches

# Convenience scripts (run all 3 datasets × 3 evidence types per LLM):
bash code/run_all_qwen.sh      # Qwen2.5-7B-Instruct-Turbo (Together AI)
bash code/run_all_gpt4_1_mini.sh   # GPT-4.1-mini (OpenAI)
bash code/run_all_llama70b.sh  # Llama-3.3-70B-Instruct-Turbo (Together AI)

# Or run individual cells directly:
# Default LLM (Qwen, no env var needed):
python code/run_main.py --dataset nq --evidence cross_encoder
python code/run_main.py --dataset triviaqa --evidence cross_encoder
python code/run_main.py --dataset popqa --evidence cross_encoder

# Switch LLM via environment variable:
export LLM_BACKEND=openai && export LLM_MODEL=gpt-4.1-mini
python code/run_main.py --dataset nq --evidence cross_encoder

export LLM_BACKEND=together && export LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
python code/run_main.py --dataset nq --evidence cross_encoder
```

`--evidence` choices used in the paper: `cross_encoder`, `embedding_stability`, `nli`, `all` (the script accepts further exploratory choices; see `--help`).

Cached LLM answers are in `data/llm_cache_*.json`. The pipeline detects these and skips redundant API calls.

### 2. Mechanism diagnostics (Appendix O, $C_1$–$C_5$)

```bash
python code/compute_c1_answer_support_mass.py
python code/compute_c2_gold_doc_mass.py
python code/compute_c3_wrong_high_conf.py
python code/compute_c4_oracle_gap.py
python code/compute_c5_cost_normalized.py
python code/compute_weight_entropy.py          # §6.4 entropy figures
```

### 3. BM25 robustness (Appendix Q)

```bash
python code/run_bm25_pipeline.py --dataset all      # Qwen × 3 datasets
python code/run_bm25_grid_search.py                  # 24-point (β, λ) grid
```

### 4. Concat-prompt baseline (Appendix P)

```bash
# Per LLM (set LLM_BACKEND / LLM_MODEL as in §1), all 3 datasets:
python code/run_concat_baseline.py --dataset all
```

### 5. Statistical tests (McNemar + Bonferroni + Bootstrap)

```bash
python code/verify_unchecked_numbers.py        # Re-verify selected derived claims
                                               # (disagreement/flip counts, McNemar
                                               # table, SmoothECE tallies; see the
                                               # script docstring for the exact list)
python code/compute_simw_es_mcnemar.py         # SimW / Dir-ES vs Naive (Appendix D)
python code/compute_nli_mcnemar.py             # Dir-NLI vs Naive, 9 cells + 216-run grid
                                               # (Appendix C, Table 14; reads the
                                               # results/*_nli_voting_*.json produced by §1)
```

### 6. Revision analyses: has-gold split, E5 retriever, aggregation rules

These produce §5.2, §6.2, and Appendices R/S/T of the paper. Canonical
outputs are bundled under `results/` for direct comparison; the
reaggregation scripts below reproduce them exactly from the per-document
answer caches (no new LLM calls).

Has-gold / no-gold decomposition (Table 3, Appendix R):

```bash
python code/compute_gold_subset_analysis.py
```

E5 retriever evaluation (Appendix S). Steps 1–2 are the *Data Download →
E5 retrieval* steps above (GPU host + Wikipedia corpus); steps 3–5 run
from the produced JSONs:

```bash
python code/run_e5_llm_pairs.py --model qwen --retriever e5   # 3. per-doc answers; repeat for gpt, llama
python code/compute_gold_subset_analysis.py --retriever e5    # 4. E5 results (Table 28)
python code/run_e5_grid_search.py                             # 5. (β,λ) grid (Table 29)
```

`run_e5_grid_search.py` includes a sanity gate: its (β=0.5, λ=30) cell and
the Naive/SimW/EO-CE baselines must match `results/gold_subset_analysis_e5.json`
to within $10^{-6}$, or it exits with an error.

Aggregation-rule comparison (Appendix T; cache reaggregation only):

```bash
python code/compute_aggregation_baselines.py                  # RRF / Borda / CE-Top1 / CE-Rerank (Table 30)
```

---

## Key Configuration

Default hyperparameters (used uniformly across all main-text cells):

```python
BETA = 0.5       # Prior strength
LAMBDA = 30.0    # Evidence strength
K = 10           # Top-k retrieved docs
```

CE: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sigmoid-applied)
ES: `sentence-transformers/all-MiniLM-L6-v2` with σ=0.01 noise, 10 perturbations
NLI: `cross-encoder/nli-deberta-v3-base` (P(entailment) + 0.5 · P(neutral))
E5: `intfloat/e5-base-v2` (raw cosine, `"query: "` / `"passage: "` prefixes; Appendix S)

---

## Statistical Methodology

- **McNemar's test**: Yates continuity correction, $\chi^2 = (|n_{01}-n_{10}|-1)^2 / (n_{01}+n_{10})$, $p$ via `scipy.stats.chi2.sf`
- **Bonferroni correction**: $\alpha/27$ (9 cells × 3 evidence signals = 27 comparisons)
- **Bootstrap CI**: 1,000 resamples, fixed seed (42) for reproducibility
- **Repeated split stability**: 3 seeds (42, 123, 456), 50/50 cal/test partition

---

## Hardware & Cost

All local orchestration, evidence scoring, and analysis ran on a consumer-grade
laptop (36 GB RAM); LLM generation used provider-hosted APIs (Together AI,
OpenAI); E5 corpus encoding used a one-time rented single-GPU host:
- Total API cost: ~$100 (Together AI + OpenAI) + ~$200 for the E5 re-run (GPU ~$10 + LLM API ~$190)
- Per-query latency: CE ~20 ms, ES ~160 ms, NLI ~110 ms
- Weight computation: <1 ms (NumPy)

---

## Citation

```bibtex
@inproceedings{seo2026retrieval,
  title     = {When Retrieval Similarity Hurts: A Diagnostic Framework for Document Weighting in Black-Box {RAG}},
  author    = {Seo, Dongju and Jang, Beakcheol},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

---

## License

Apache 2.0 (see LICENSE file).

Datasets, models, and other artifacts retain their original licenses (see paper Appendix J for details).
