# When Retrieval Similarity Hurts: A Diagnostic Framework for Document Weighting in Black-Box RAG

Code and result files for the paper

> Dongju Seo and Beakcheol Jang. **When Retrieval Similarity Hurts: A Diagnostic Framework for Document Weighting in Black-Box RAG.** *Findings of the Association for Computational Linguistics: EMNLP 2026.*

- Paper: Findings of the Association for Computational Linguistics: EMNLP 2026 (ACL Anthology entry to appear)
- Cached LLM outputs and evidence caches (≈98 MB compressed / ≈289 MB extracted): [Zenodo, DOI `10.5281/zenodo.22054839`](https://doi.org/10.5281/zenodo.22054839). With the caches and the retrieval inputs described under *Data Download*, every number in the paper is reproducible from this code **without any new API calls**.

We propose a unified two-parameter weighting framework that generalizes REPLUG by adding an evidence term, with retrieval-similarity weighting ($\lambda{=}0$) and evidence-only weighting ($\lambda{\to}\infty$) emerging as limiting cases.

---

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── requirements-lock.txt              # pip freeze of the authors' environment (exact versions)
├── input_sha256sums.txt               # SHA-256 of the retrieval input files (shasum -a 256 -c)
├── LICENSE                            # Apache 2.0
├── .env.example                       # Environment-variable template
├── .gitignore                         # Standard ignores (env, caches, large data)
├── code/                              # All experiment scripts (39 .py + 3 .sh)
│   ├── weighting.py                  # Core framework (Eq. 2)
│   ├── metrics.py                    # McNemar, F1, EM, AUROC
│   ├── generation.py                 # Black-box LLM API caller (with answer cache)
│   ├── evidence_scores.py            # CE / ES / NLI evidence
│   ├── precompute_evidence_cache.py  # Rebuild data/evidence_cache/ (CE + ES)
│   ├── run_main.py                   # Main 9-cell pipeline (Table 2; per-query EM vectors)
│   ├── regenerate_table2.py          # Table 2 rows: full-test point estimates + 3-seed ±std
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
    │                                   #   PopQA rows use the full set n=14,267;
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
    │                                   #   strict k=10 cache re-aggregation; full-bucket EM
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
repository. Three large artifact groups are not versioned in Git (groups 1 and 2
are archived on Zenodo; group 3 is regenerated locally):

1. **Cached LLM outputs** (`data/llm_cache_*.json`: ≈116 MB for the 3 LLMs ×
   31,861 queries × top-$k$ docs on DPR/Contriever, ≈110 MB for the E5
   re-run, ≈27 MB for the BM25 and ≈8 MB for the concat-prompt baseline
   caches, plus an ≈8 MB NQ-only Meta-Llama-3.1-8B-Instruct-Turbo pilot cache
   that is not used in the paper). These let every pipeline skip API calls.
2. **Evidence caches** (`data/evidence_cache/`, ≈20 MB): per-query CE/ES
   scores consumed by the analysis scripts. Also regenerable deterministically
   with local models only: `python code/precompute_evidence_cache.py`
   (E5 variants via `assemble_e5_and_ce.py`).
3. **Per-query voting JSONs** (`results/*_voting_*.json`, ≈75 MB across 27
   (LLM × dataset × evidence) cells, including the `_per_question` EM vectors
   used by the McNemar scripts). Regenerated by `run_main.py` from the caches
   above (no API calls; only the local evidence models run).

Groups 1–2 are archived on **Zenodo** ([`10.5281/zenodo.22054839`](https://doi.org/10.5281/zenodo.22054839)) as three tarballs
(`llm_caches_dpr_contriever.tar.gz`, `llm_caches_e5.tar.gz`,
`evidence_caches.tar.gz`; sha256 in `SHA256SUMS.txt` and a per-file manifest).
Extract them into `data/` at the package root (a fresh clone has no `data/`
directory; create it first):

```bash
mkdir -p data
tar xzf llm_caches_dpr_contriever.tar.gz -C data/
tar xzf llm_caches_e5.tar.gz -C data/
tar xzf evidence_caches.tar.gz -C data/     # creates data/evidence_cache/
```

The archives store numeric owner 0:0 and carry no extended attributes, so they
extract cleanly for any user, including root in a container.

---

## Installation

```bash
pip install -r requirements.txt
```

Developed on macOS; Linux should work as well. The analysis and re-aggregation scripts in this release were last run with Python 3.13/3.14. Package versions installed in the authors' environment at release time (August 2026): numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, torch 2.11.0, transformers 5.4.0, sentence-transformers 5.3.0, pyserini 2.0.0, bm25s 0.3.6, datasets 4.8.4, openai 2.32.0, pandas 3.0.1, matplotlib 3.10.8. The released LLM caches remove every LLM API call. CE and ES evidence is shipped in the evidence caches (used by the analysis scripts) and is also recomputed locally by `run_main.py`; NLI evidence is not cached and is always recomputed with the local `cross-encoder/nli-deberta-v3-base` model (about 0.11 s per query). The summary JSONs in `results/` hold the reported values. `requirements-lock.txt` is the `pip freeze` of the authors' environment; input-file checksums and model revisions are listed under *Pinned inputs and model revisions* below.

---

## Data Download

We exclude raw test sets due to size and licensing. Reproduce by downloading:

### Required public datasets

1. **NQ-Open** (CC-BY-SA 3.0):
   - Download from https://github.com/google-research-datasets/natural-questions
   - We use the 8,757 dev questions of the DPR split (as released by ReConsider; see *Retrieval inputs* below).

2. **TriviaQA** (Apache 2.0):
   - Download from https://nlp.cs.washington.edu/triviaqa/
   - We use the 8,837 dev questions of the DPR split (as released by ReConsider; see *Retrieval inputs* below).

3. **PopQA** (MIT):
   - Download from https://huggingface.co/datasets/akariasai/PopQA
   - We use 14,267 queries.

### Retrieval inputs (`data/nq_cosine.json`, `data/triviaqa_cosine.json`, `data/popqa_contriever.json`)

The pipeline reads three retrieval files that we do not redistribute (they
contain Wikipedia passage text and third-party retrieval outputs). Rebuild
them as follows. The LLM caches are keyed by
`md5(model ||| question ||| first 800 characters of the document text)`, so
the rebuilt documents must match the original sources character for
character; use exactly the files named here.

**NQ-Open and TriviaQA (DPR, top-10).** Download the DPR retriever outputs
released by ReConsider (Iyer et al., 2021; multiset DPR encoders, dev splits)
and convert them:

```bash
mkdir -p data
wget http://dl.fbaipublicfiles.com/reconsider/dpr_retriever_outputs/nq-dev-multi.json  -O data/nq_dpr_dev.json
wget http://dl.fbaipublicfiles.com/reconsider/dpr_retriever_outputs/tqa-dev-multi.json -O data/tqa_dpr_dev.json
python code/convert_dpr_cosine.py     # writes data/nq_cosine.json and data/triviaqa_cosine.json
```

Each downloaded file is JSON Lines with one object per question (`id`,
`question`, `answers`, `gt_title`, `positives`, `negatives`); every entry of
`positives` is `[title, tokens, answer_spans]` for one retrieved passage
(up to 100 per question). `convert_dpr_cosine.py` keeps the first 10 distinct
passages per question (text = `title. text`, deduplicated on the case-folded
first 200 characters), re-encodes questions and passages with
`facebook/dpr-question_encoder-multiset-base` and
`facebook/dpr-ctx_encoder-multiset-base`, and stores the cosine similarity as
`score`. Output schema: `{question, answers, retrieved_docs: [{text, type,
score}]}` with 8,757 NQ and 8,837 TriviaQA questions. We downloaded the two
files on April 1, 2026; the ReConsider repository is released under CC-BY-NC.

**PopQA (Contriever, top-10).** Questions are loaded from the Hugging Face
dataset `akariasai/PopQA` (14,267 questions). Passages are the DPR Wikipedia
split, scored with the pre-computed Contriever-MSMARCO passage embeddings:

```bash
mkdir -p data
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -O data/psgs_w100.tsv.gz && gunzip data/psgs_w100.tsv.gz
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar -O data/wikipedia_embeddings.tar
tar xf data/wikipedia_embeddings.tar -C data/   # unpacks to data/wikipedia_embeddings/passages_00 ... passages_15 (16 shards, about 32 GB)
ls data/wikipedia_embeddings/passages_* | wc -l  # expect 16
python code/build_popqa_contriever.py            # writes data/popqa_contriever.json
```

`build_popqa_contriever.py` encodes each question with
`facebook/contriever-msmarco` and L2-normalizes question and passage vectors.
It takes the top-10 passages by dot product over all 16 shards (shard by
shard, then merged; `score` is the dot product rounded to 6 decimals) and
attaches the passage text from `psgs_w100.tsv` (`title. text`, deduplicated
on the case-folded first 200 characters). Each document is labelled by
answer-string matching over answers of three or more characters: the first
answer-bearing passage is `gold`, later ones `related`, the rest `noise`.
PopQA records additionally carry `s_pop` (subject popularity from PopQA).
The script downloads `psgs_w100.tsv` itself if the file is missing. We built
the PopQA retrieval set on April 4, 2026.

**Pinned inputs and model revisions.** The LLM caches are keyed on the exact
document text, so the rebuilt inputs must match ours byte for byte.
`input_sha256sums.txt` lists the SHA-256 of the downloaded files as we used
them (the two ReConsider files, `psgs_w100.tsv` after `gunzip`, and the 16
embedding shards as extracted); check with
`shasum -a 256 -c input_sha256sums.txt` from the package root. The Hugging
Face revisions resolved when we downloaded the models and the dataset were:

| Artifact | Revision (commit) |
|---|---|
| `facebook/dpr-question_encoder-multiset-base` | `5325e4ee906435291d63046f535476cb3fc60d43` |
| `facebook/dpr-ctx_encoder-multiset-base` | `fdb3d46584386d2f20aa00724ae31cebc348d16b` |
| `facebook/contriever-msmarco` | `abe8c1493371369031bcb1e02acb754cf4e162fa` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | `c5ee24cb16019beea0893ab7796b1df96625c6b8` (the repository, now `cross-encoder/ms-marco-MiniLM-L6-v2`, was updated on 2026-08-09, after our runs) |
| `sentence-transformers/all-MiniLM-L6-v2` | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` (main runs, April 2026) and `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (E5 runs, July 2026); both ship the same `model.safetensors` |
| `cross-encoder/nli-deberta-v3-base` | `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` |
| `intfloat/e5-base-v2` | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| `roberta-large-mnli` (alternative NLI model) | `2a8f12d27941090092df78e4ba6f0928eb5eac98` |
| `akariasai/PopQA` (dataset; `test` split, 14,267 rows) | `098765c79ea10a2cb19c828324e33281b8336ec0` |

Set `PIN_MODEL_REVISIONS=1` (a shell `export`; the pipeline does not read
`.env`) to pass these commits to each loader of the versioned external
artifacts listed above (`MODEL_REVISIONS` in `code/config.py`; the E5 encoding
script carries its own copy for the GPU host; the optional local HF LLM and
utility-predictor loaders are not covered). By default the code
loads the Hub head, which on 2026-08-22 still resolved to the listed commit for
every artifact except the cross-encoder. For the three `facebook/*` encoders
`transformers` also fetched the safetensors conversion of the same checkpoint
from each model's conversion pull request; we verified tensor by tensor that it
equals `pytorch_model.bin` at the commit above. `requirements-lock.txt` is the `pip freeze` of the
authors' environment (macOS 26.5, Apple Silicon, Python 3.14.3) at release
time; `pip install -r requirements-lock.txt` reproduces that package set,
`requirements.txt` gives the minimal one.

The BM25 scripts (Appendix Q) read `data/nq_test.json` / `data/triviaqa_test.json`:
these are the same DPR dev-split files as `data/nq_cosine.json` /
`data/triviaqa_cosine.json` (identical schema; only `question`/`answers` are used
for BM25; copy or symlink the cosine files under the `*_test.json` names).

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
export TOGETHER_API_KEY=<your_key>      # Qwen, Llama (Together AI); not needed with the caches
export OPENAI_API_KEY=<your_key>         # GPT-4.1-mini              ; not needed with the caches
export PIN_MODEL_REVISIONS=1             # load the recorded model/dataset commits (see "Pinned inputs and model revisions")

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

`--evidence` choices used in the paper: `cross_encoder`, `embedding_stability`, and `nli` (the convenience scripts loop over exactly these three). `--evidence all` also runs the exploratory methods listed in `config.EVIDENCE_METHODS` (`response_confidence`, `llm_judge`, `utility_predictor`), which need a local HF model or a trained utility predictor; it is not a paper-reproduction shortcut.

`python code/regenerate_table2.py` rebuilds the Table 2 rows from the voting JSONs written above (full-test point estimates) and from the repeated-split JSONs (±std; the split mean/std pairs are also written to `results/table1_mean_std.json`).

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

### 5. Statistical tests (McNemar + Bonferroni)

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
- `metrics.bootstrap_ci` is an unused utility (imported by `run_main.py`, never called); no bootstrap interval is reported in the paper
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
