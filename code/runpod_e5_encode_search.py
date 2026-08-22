#!/usr/bin/env python3
"""
Modern dense retriever (E5 / BGE) corpus encoding + top-10 retrieval.
Robustness check with modern dense retrievers (paper Appendix S).

Runs on a RunPod A100-80GB pod (Secure Cloud). Encodes the full ~21M-passage DPR
Wikipedia corpus (psgs_w100) with a modern dense retriever, builds a FAISS
IndexFlatIP, and retrieves top-10 for each NQ / TriviaQA / PopQA query. Emits a
compact per-query [[pid, score], ...] file for LOCAL assembly (script 3), so the
13 GB corpus is never shipped back.

TWO MODES
---------
  prep  : LOCAL. Extract {question, answers} from the existing *_cosine files into
          a lightweight {dataset}_queries.json to upload to the pod.
  run   : POD. Download corpus (if absent), encode, index, search, diagnose, save.

CRITICAL — instruction prefixes. Omitting them silently destroys retrieval
quality (no error, just bad results), so they are applied explicitly here:
  e5-base-v2        query "query: "                                             passage "passage: "
  bge-base-en-v1.5  query "Represent this sentence for searching relevant passages: "  passage ""
Both models are L2-normalized (cosine == inner product), matching weighting.py's
assumption that similarities live on the same [-1, 1] scale as the DPR cosines.

Text format matches convert_dpr_cosine.py / build_popqa_contriever.py exactly:
  passage text = f"{title}. {text}" from psgs_w100.tsv (parts[0]=pid, [1]=text, [2]=title).

Determinism: encoders run in eval mode (no dropout); FAISS IndexFlatIP is exact.
np seed fixed. Re-running yields identical retrieval.

USAGE
-----
  # locally:
  python runpod_e5_encode_search.py prep
  #   -> data/e5_queries_{nq,triviaqa,popqa}.json ; upload these + this script to the pod

  # on the pod (after `pip install sentence-transformers faiss-cpu tqdm`):
  # POD SIZING: IndexFlatIP holds 21M x 768 x 4B ~= 65 GB in host RAM -> provision an
  # A100-80GB pod with >= 120 GB system RAM or it OOMs mid-index.
  python runpod_e5_encode_search.py run --model e5  --datasets nq triviaqa popqa
  python runpod_e5_encode_search.py run --model bge --datasets nq triviaqa popqa
  #   -> {dataset}_{model}_search.json ; download these back to local
"""
import argparse
import gc
import json
import os
import subprocess
import sys

import numpy as np

SEED = 42
np.random.seed(SEED)

# Determinism for the pod's CUDA encoder (global rule: fixed seed + deterministic
# kernels). Encoding is forward-only/eval so already near-deterministic; pinning the
# cuBLAS workspace + deterministic algorithms guarantees bit-identical retrieval on
# re-run. Must be set before torch is imported (torch is lazy-imported in load_model).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

# On the pod, DATA_DIR may not exist; allow override via env.
DATA_DIR = os.environ.get("E5_DATA_DIR", DATA_DIR)

CORPUS_TSV = os.path.join(DATA_DIR, "psgs_w100.tsv")
CORPUS_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"

MODELS = {
    "e5": {
        "hf_name": "intfloat/e5-base-v2",
        "q_prefix": "query: ",
        "p_prefix": "passage: ",
    },
    "bge": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "q_prefix": "Represent this sentence for searching relevant passages: ",
        "p_prefix": "",
    },
}

# Source files to extract queries from (existing retrieval sets; reuse their
# exact question/answer strings so query counts match the paper: 8757/8837/14267).
QUERY_SOURCES = {
    "nq":       os.path.join(DATA_DIR, "nq_cosine.json"),
    "triviaqa": os.path.join(DATA_DIR, "triviaqa_cosine.json"),
    "popqa":    os.path.join(DATA_DIR, "popqa_contriever.json"),
}

NUM_DOCS = 10           # final docs/query after local dedup (script 3)
SEARCH_K = 15           # retrieve extra so local dedup can still reach NUM_DOCS
ENCODE_BATCH = 1024     # A100 80GB with fp16; lower if VRAM-bound
CORPUS_CHUNK = 500_000  # passages encoded+added per chunk (bounds peak host RAM)


# ============================================================
# prep (LOCAL)
# ============================================================

def prep():
    """Extract lightweight {question, answers} lists for pod upload."""
    for ds, src in QUERY_SOURCES.items():
        if not os.path.exists(src):
            print(f"  [skip] {ds}: source not found ({src})")
            continue
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = [{"question": s["question"], "answers": s["answers"]} for s in data]
        out = os.path.join(DATA_DIR, f"e5_queries_{ds}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False)
        print(f"  {ds}: {len(queries)} queries -> {out}")
    print("prep done. Upload e5_queries_*.json + this script to the pod.")


# ============================================================
# run (POD)
# ============================================================

def download_corpus():
    if os.path.exists(CORPUS_TSV):
        print(f"  corpus present: {CORPUS_TSV}")
        return
    print("  downloading psgs_w100.tsv.gz (~4.6 GB)...")
    subprocess.run(["curl", "-L", "-o", CORPUS_TSV + ".gz", CORPUS_URL], check=True)
    subprocess.run(["gunzip", CORPUS_TSV + ".gz"], check=True)


def iter_corpus():
    """Yield (pid, raw_passage_text) with raw = 'title. text' (no model prefix)."""
    with open(CORPUS_TSV, "r", encoding="utf-8") as f:
        f.readline()  # header: id\ttext\ttitle
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                pid, text, title = parts[0].strip(), parts[1], parts[2]
                yield pid, (f"{title}. {text}" if title else text)


def load_model(model_key):
    from sentence_transformers import SentenceTransformer
    import torch
    # Determinism (global rule). warn_only=True: if a specific encoder kernel lacks a
    # deterministic implementation it warns instead of raising, so encoding still runs.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading {cfg['hf_name']} on {device}")
    model = SentenceTransformer(cfg["hf_name"], device=device)
    if device == "cuda":
        # fp16 on A100 TensorCores: encoding is the bottleneck and fp32 leaves the
        # TensorCores idle (~5-10x slower). Embeddings are cast back to float32 for
        # FAISS; fp16 encoding is standard practice and does not affect retrieval.
        model = model.half()
        print("  using fp16 (A100 TensorCore)")
    return model, cfg


def build_index(model, cfg):
    """Stream-encode the corpus into a FAISS IndexFlatIP; return (index, pids)."""
    import faiss
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)  # exact cosine (embeddings are L2-normalized)
    pids = []

    chunk_pids, chunk_texts = [], []
    total = 0

    def flush():
        nonlocal chunk_pids, chunk_texts, total
        if not chunk_texts:
            return
        embs = model.encode(
            [cfg["p_prefix"] + t for t in chunk_texts],
            batch_size=ENCODE_BATCH, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        index.add(embs)
        pids.extend(chunk_pids)
        total += len(chunk_texts)
        print(f"    indexed {total:,} passages", flush=True)
        chunk_pids, chunk_texts = [], []
        del embs
        gc.collect()

    for pid, text in iter_corpus():
        chunk_pids.append(pid)
        chunk_texts.append(text)
        if len(chunk_texts) >= CORPUS_CHUNK:
            flush()
    flush()

    print(f"  index built: {index.ntotal:,} vectors, dim={dim}")
    return index, pids


def search_dataset(model, cfg, index, pids, ds, model_key):
    qpath = os.path.join(DATA_DIR, f"e5_queries_{ds}.json")
    with open(qpath, "r", encoding="utf-8") as f:
        queries = json.load(f)
    questions = [q["question"] for q in queries]

    q_embs = model.encode(
        [cfg["q_prefix"] + q for q in questions],
        batch_size=ENCODE_BATCH, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)

    scores, idxs = index.search(q_embs, SEARCH_K)  # [Q, SEARCH_K]

    results = []
    for qi in range(len(questions)):
        results.append([[pids[idxs[qi][r]], float(scores[qi][r])]
                        for r in range(SEARCH_K)])

    out = os.path.join(DATA_DIR, f"{ds}_{model_key}_search.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f)

    diagnose(scores, ds, model_key)
    print(f"  saved {out}  ({len(results)} queries x top-{SEARCH_K})")


def diagnose(scores, ds, model_key):
    """Cosine-distribution sanity check (mirrors convert_dpr_cosine.py:187-196).

    Written to BOTH stdout and {ds}_{model}_diag.txt so an unattended A100 batch
    still leaves a permanent record. If the distribution is far
    from the DPR band (mean ~0.66, per-query top-10 spread ~0.06), inspect before
    trusting downstream EM: a collapsed spread + a prefix bug look identical to
    'the retriever is just very confident'. The fixed paper defaults (beta=0.5,
    lambda=30) are applied to whatever scale this reports — no per-retriever tuning,
    which is the conservative/fair choice; report the scale for transparency.
    """
    flat = scores.reshape(-1)
    top1 = scores[:, 0]
    spread = scores[:, 0] - scores[:, NUM_DOCS - 1]  # per-query top1 - top10
    lines = [
        f"  [{ds} / {model_key}] cosine diagnostics:",
        f"    all: mean={flat.mean():.4f} std={flat.std():.4f} "
        f"min={flat.min():.4f} max={flat.max():.4f}",
        f"    top1: mean={top1.mean():.4f}   per-query top1-top10 spread: "
        f"mean={spread.mean():.4f} median={np.median(spread):.4f}",
    ]
    for beta in (0.5, 1.0, 2.0, 4.0):
        lines.append(f"    exp(beta*max) beta={beta}: {np.exp(beta * flat.max()):.2f}")
    for ln in lines:
        print(ln)
    with open(os.path.join(DATA_DIR, f"{ds}_{model_key}_diag.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(model_key, datasets):
    download_corpus()
    model, cfg = load_model(model_key)
    index, pids = build_index(model, cfg)
    for ds in datasets:
        search_dataset(model, cfg, index, pids, ds, model_key)
    print("run done. Download *_search.json back to local for assembly (script 3).")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("prep", help="LOCAL: extract query lists for pod upload")
    r = sub.add_parser("run", help="POD: encode corpus, index, search")
    r.add_argument("--model", choices=list(MODELS), default="e5")
    r.add_argument("--datasets", nargs="+", default=["nq", "triviaqa", "popqa"],
                   choices=list(QUERY_SOURCES))
    a = p.parse_args()

    if a.mode == "prep":
        prep()
    else:
        run(a.model, a.datasets)


if __name__ == "__main__":
    main()
