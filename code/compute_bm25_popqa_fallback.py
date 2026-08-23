"""Appendix Q: PopQA BM25 EM on all 14,267 queries vs. the queries unaffected by the
API-failure fallbacks of the original run (per-document calls that failed on every retry
are cached as the abstention token "unknown" and receive no vote).

Uses run_bm25_pipeline's own helpers (retrieval/LLM/CE caches, vote rule), so the 'all'
row reproduces results/bm25_pipeline_qwen_popqa.json exactly.
Output: results/bm25_popqa_fallback_subset.json
"""
import sys, os, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bm25_pipeline as rb
from metrics import normalize_answer
from weighting import naive_weights, replug_weights, dirichlet_weights
rb.load_llm_cache(); llm = rb._llm_cache; ce_cache = json.load(open(rb.BM25_CE_CACHE))
test = json.load(open(os.path.join(rb.DATA_DIR, rb.DATASET_FILES['popqa'])))
qs = [s['question'] for s in test]; retrieval = rb.bm25_retrieve_all(qs, k=rb.NUM_DOCS)
ids = sorted({h['id'] for q in qs for h in retrieval.get(q, [])[:rb.NUM_DOCS]}); text = {i: rb.fetch_doc_text(i) for i in ids}
rows = []
for s in test:
    q = s['question']; gold = s.get('answers', []); hits = retrieval.get(q, [])[:rb.NUM_DOCS]
    if not gold or len(hits) != rb.NUM_DOCS: continue
    texts = [text.get(h['id'], '') for h in hits]; answers = [llm.get(rb.llm_cache_key(q, t), 'unknown') for t in texts]
    n_fb = sum(a == 'unknown' for a in answers)
    raw = [h['score'] for h in hits]; smin, smax = min(raw), max(raw); sims = [(x - smin) / (smax - smin) for x in raw] if smax > smin else [0.5] * len(raw)
    evs = [ce_cache[rb.ce_cache_key(q, t)] for t in texts]
    W = {'naive': naive_weights(rb.NUM_DOCS), 'simw': replug_weights(sims, beta=rb.BETA), 'dir_ce': dirichlet_weights(sims, evs, beta=rb.BETA, lam=rb.LAMBDA_DIR_CE), 'eo_ce': [e / max(sum(evs), 1e-9) for e in evs]}
    gold_norm = {normalize_answer(g) for g in gold if g}; em = {}
    for m, w in W.items():
        vote = defaultdict(float)
        for a, wi in zip(answers, w):
            nm = normalize_answer(a)
            if nm and nm != 'unknown': vote[nm] += wi
        pred = max(vote, key=vote.get) if vote else ''
        em[m] = int(pred != '' and pred in gold_norm)  # pipeline: an all-abstention query is never counted correct
    rows.append((n_fb, em))
def report(label, sel):
    n = len(sel); out = {m: 100 * sum(r[1][m] for r in sel) / n for m in ['naive', 'simw', 'dir_ce', 'eo_ce']}
    print(f"{label}: n={n} " + ' '.join(f"{m}={v:.2f}" for m, v in out.items()) + f" | dDir={out['dir_ce']-out['naive']:+.2f} dEO={out['eo_ce']-out['naive']:+.2f}")
    return out
all_ = report('all', rows); unaff = report('no-fallback', [r for r in rows if r[0] == 0]); allfb = [r for r in rows if r[0] == 10]
print('all-fallback queries', len(allfb), 'partial', sum(1 for r in rows if 0 < r[0] < 10))
json.dump({'all': all_, 'unaffected': unaff, 'n_all': len(rows), 'n_unaffected': sum(1 for r in rows if r[0] == 0), 'n_all_fallback': len(allfb)}, open(os.path.join(rb.RESULTS_DIR, 'bm25_popqa_fallback_subset.json'), 'w'), indent=2)
