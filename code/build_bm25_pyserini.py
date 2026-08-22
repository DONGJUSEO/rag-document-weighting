"""
Build BM25 index over Wikipedia psgs_w100 using Pyserini (Lucene backend).

Pyserini's Java/Lucene backend is memory-efficient for 21M passages,
unlike pure-Python bm25s which OOMs.

Workflow:
  1. Convert psgs_w100.tsv → jsonl shards (streaming, low memory)
  2. Run pyserini.index.lucene CLI to build index

Usage:
    cd code/
    python3 build_bm25_pyserini.py [--skip-jsonl]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

CORPUS_TSV = os.path.join(DATA_DIR, "psgs_w100.tsv")
JSONL_DIR = os.path.join(DATA_DIR, "psgs_jsonl")
INDEX_DIR = os.path.join(DATA_DIR, "bm25_pyserini_index")

SHARD_SIZE = 1_000_000  # passages per shard


def convert_tsv_to_jsonl():
    """Stream TSV → JSONL shards. Low-memory."""
    os.makedirs(JSONL_DIR, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    n = 0
    shard_idx = 0
    shard_path = os.path.join(JSONL_DIR, f"shard_{shard_idx:03d}.jsonl")
    out_f = open(shard_path, "w", encoding="utf-8")

    t0 = time.time()
    print(f"[1/2] Converting TSV → JSONL shards ({SHARD_SIZE:,} per shard)...")

    with open(CORPUS_TSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            doc = {
                "id": row.get("id", str(n)),
                "contents": f"{row.get('title', '').strip()} {row.get('text', '').strip()}".strip(),
            }
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1

            if n % SHARD_SIZE == 0:
                out_f.close()
                elapsed = time.time() - t0
                rate = n / max(elapsed, 0.001)
                print(f"  Shard {shard_idx:03d} done ({n:,} total, {rate:.0f}/s)", flush=True)
                shard_idx += 1
                shard_path = os.path.join(JSONL_DIR, f"shard_{shard_idx:03d}.jsonl")
                out_f = open(shard_path, "w", encoding="utf-8")

    out_f.close()
    elapsed = time.time() - t0
    print(f"  Done: {n:,} passages → {shard_idx + 1} shards in {elapsed/60:.1f} min")


def build_index():
    """Run pyserini.index.lucene CLI."""
    print(f"\n[2/2] Building BM25 Lucene index...")
    print(f"  Input:  {JSONL_DIR}")
    print(f"  Output: {INDEX_DIR}")

    os.makedirs(INDEX_DIR, exist_ok=True)
    t0 = time.time()

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", JSONL_DIR,
        "--index", INDEX_DIR,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]

    print(f"  Cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jsonl", action="store_true",
                        help="Skip TSV→JSONL conversion (use existing shards)")
    args = parser.parse_args()

    print(f"=== BM25 Pyserini index build ===")
    print(f"Corpus:    {CORPUS_TSV}")
    print(f"JSONL dir: {JSONL_DIR}")
    print(f"Index dir: {INDEX_DIR}\n")

    if not args.skip_jsonl:
        convert_tsv_to_jsonl()
    else:
        print("[1/2] Skipping JSONL conversion (--skip-jsonl)")

    build_index()
    print(f"\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
