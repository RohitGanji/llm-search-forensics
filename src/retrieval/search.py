from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index.build_indexes import load_indexes, tokenize, ChunkMeta


Mode = Literal["bm25", "vector", "hybrid"]


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    doc_id: str
    header_path: str
    source_type: str


def search_bm25(query: str, k: int, bm25, metas: List[ChunkMeta]) -> List[SearchResult]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens).astype(np.float32)
    idx = np.argsort(-scores)[:k]
    out: List[SearchResult] = []
    for i in idx:
        m = metas[int(i)]
        out.append(
            SearchResult(
                chunk_id=m.chunk_id,
                score=float(scores[int(i)]),
                doc_id=m.doc_id,
                header_path=m.header_path,
                source_type=m.source_type,
            )
        )
    return out


def search_vector(
    query: str,
    k: int,
    faiss_index,
    metas: List[ChunkMeta],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[SearchResult]:
    model = SentenceTransformer(model_name, device="cpu")
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    scores, indices = faiss_index.search(q, k)
    out: List[SearchResult] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        m = metas[int(idx)]
        out.append(
            SearchResult(
                chunk_id=m.chunk_id,
                score=float(score),  # inner product ~ cosine (since normalized)
                doc_id=m.doc_id,
                header_path=m.header_path,
                source_type=m.source_type,
            )
        )
    return out


def search_hybrid(
    query: str,
    k: int,
    bm25,
    faiss_index,
    metas: List[ChunkMeta],
    bm25_k: int | None = None,
    vec_k: int | None = None,
) -> List[SearchResult]:
    """
    Simple hybrid:
      - retrieve bm25_k and vec_k candidates
      - normalize scores
      - sum weighted scores
      - return top-k
    """
    bm25_k = bm25_k or max(k * 5, 50)
    vec_k = vec_k or max(k * 5, 50)

    bm = search_bm25(query, bm25_k, bm25, metas)
    ve = search_vector(query, vec_k, faiss_index, metas)

    # normalize per-retriever (min-max); robust enough for MVP
    def minmax(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"min": 0.0, "max": 1.0}
        return {"min": float(min(xs)), "max": float(max(xs))}

    bm_stats = minmax([r.score for r in bm])
    ve_stats = minmax([r.score for r in ve])

    def norm(x: float, mn: float, mx: float) -> float:
        if mx <= mn:
            return 0.0
        return (x - mn) / (mx - mn)

    merged: Dict[str, SearchResult] = {}

    # weights (tunable later)
    w_bm, w_ve = 0.45, 0.55

    for r in bm:
        s = w_bm * norm(r.score, bm_stats["min"], bm_stats["max"])
        merged[r.chunk_id] = SearchResult(r.chunk_id, s, r.doc_id, r.header_path, r.source_type)

    for r in ve:
        s = w_ve * norm(r.score, ve_stats["min"], ve_stats["max"])
        if r.chunk_id in merged:
            prev = merged[r.chunk_id]
            merged[r.chunk_id] = SearchResult(r.chunk_id, prev.score + s, r.doc_id, r.header_path, r.source_type)
        else:
            merged[r.chunk_id] = SearchResult(r.chunk_id, s, r.doc_id, r.header_path, r.source_type)

    out = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:k]
    return out


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "vector", "hybrid"])
    p.add_argument("--index_dir", type=str, default="indexes")
    args = p.parse_args()

    bm25, vindex, metas = load_indexes(args.index_dir)

    if args.mode == "bm25":
        res, dt = timed(search_bm25, args.query, args.k, bm25, metas)
    elif args.mode == "vector":
        res, dt = timed(search_vector, args.query, args.k, vindex, metas)
    else:
        res, dt = timed(search_hybrid, args.query, args.k, bm25, vindex, metas)

    print(json.dumps({"mode": args.mode, "k": args.k, "latency_s": dt, "results": [r.__dict__ for r in res]}, indent=2))


if __name__ == "__main__":
    main()
