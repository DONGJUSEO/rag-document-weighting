"""
Build BM25 index over Wikipedia psgs_w100 corpus (21M passages).

Uses bm25s (pure Python + NumPy, fast).

Usage:
    cd code/
    python3 build_bm25_index.py

Output:
    data/bm25_wiki_index/  (BM25 index files)
"""

import os
import sys
import time
import csv

import bm25s
import Stemmer  # PyStemmer

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

CORPUS_FILE = os.path.join(DATA_DIR, "psgs_w100.tsv")
INDEX_DIR = os.path.join(DATA_DIR, "bm25_wiki_index")


def stream_corpus_text(tsv_path, log_every=1_000_000):
    """Yield passage text only (id, title discarded for index)."""
    csv.field_size_limit(sys.maxsize)
    n = 0
    t0 = time.time()
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n += 1
            text = row.get("text", "").strip()
            title = row.get("title", "").strip()
            # Index "title text" (matches DPR convention)
            yield f"{title} {text}".strip()
            if n % log_every == 0:
                elapsed = time.time() - t0
                rate = n / max(elapsed, 0.001)
                print(f"  Loaded {n:,} passages ({rate:.0f}/s, {elapsed:.0f}s elapsed)", flush=True)


def main():
    print(f"=== BM25 Wikipedia index build ===")
    print(f"Corpus: {CORPUS_FILE}")
    print(f"Output: {INDEX_DIR}")
    print()

    # 1. Read all passages into memory (text only)
    print(f"[1/3] Loading corpus...")
    t0 = time.time()
    corpus = list(stream_corpus_text(CORPUS_FILE))
    n = len(corpus)
    print(f"  Loaded {n:,} passages in {(time.time()-t0)/60:.1f} min")
    print()

    # 2. Tokenize
    print(f"[2/3] Tokenizing (English stopwords + Porter stemmer)...")
    t0 = time.time()
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(
        corpus,
        stopwords="en",
        stemmer=stemmer,
        show_progress=True,
    )
    print(f"  Tokenized in {(time.time()-t0)/60:.1f} min")
    print()

    # 3. Build BM25 index
    print(f"[3/3] Building BM25 index...")
    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=True)
    print(f"  Indexed in {(time.time()-t0)/60:.1f} min")
    print()

    # 4. Save
    os.makedirs(INDEX_DIR, exist_ok=True)
    retriever.save(INDEX_DIR)
    print(f"=== Saved to {INDEX_DIR} ===")

    # Memory hint: corpus list freed after this scope


if __name__ == "__main__":
    main()
