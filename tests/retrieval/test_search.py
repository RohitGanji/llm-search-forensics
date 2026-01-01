import os
from src.index.build_indexes import load_indexes
from src.retrieval.search import search_bm25, search_vector, search_hybrid

def test_search_returns_k_results():
    # requires indexes built
    assert os.path.exists("indexes/meta.json"), "Run: uv run python -m src.index.build_indexes"
    bm25, vindex, metas = load_indexes("indexes")

    q = "permissions sharing public link"
    k = 5

    bm = search_bm25(q, k, bm25, metas)
    ve = search_vector(q, k, vindex, metas)
    hy = search_hybrid(q, k, bm25, vindex, metas)

    assert len(bm) == k
    assert len(ve) == k
    assert len(hy) == k

    assert all(r.chunk_id for r in hy)
    assert all(isinstance(r.score, float) for r in hy)
