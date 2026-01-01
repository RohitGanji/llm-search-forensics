import os
import json
import pickle
import numpy as np

from src.index.build_indexes import (
    build_bm25,
    build_faiss_index,
    save_indexes,
    load_indexes,
    ChunkMeta,
)

def test_bm25_build_and_score_smoke():
    texts = [
        "Slack workflow webhook trigger external system",
        "Notion sharing permissions public links",
        "Confluence page permissions restrict access",
    ]
    bm25, tokenized = build_bm25(texts)
    assert len(tokenized) == 3
    scores = bm25.get_scores(["permissions"])
    assert len(scores) == 3
    assert float(np.max(scores)) >= 0.0

def test_faiss_index_build_smoke():
    texts = [
        "create jira automation rule",
        "trigger slack workflow webhook",
        "notion permissions sharing",
    ]
    index, emb = build_faiss_index(texts, batch_size=8)
    assert emb.shape[0] == 3
    assert index.ntotal == 3

def test_save_and_load_indexes(tmp_path):
    texts = [
        "alpha permissions",
        "beta workflow",
        "gamma automation",
    ]
    bm25, _ = build_bm25(texts)
    index, _ = build_faiss_index(texts, batch_size=8)

    metas = [
        ChunkMeta(chunk_id="c0", doc_id="d0", header_path="H1", source_type="html"),
        ChunkMeta(chunk_id="c1", doc_id="d1", header_path="H1 > H2", source_type="html"),
        ChunkMeta(chunk_id="c2", doc_id="d2", header_path="", source_type="html"),
    ]

    out_dir = str(tmp_path / "idx")
    save_indexes(out_dir, bm25, index, metas)

    assert os.path.exists(os.path.join(out_dir, "bm25.pkl"))
    assert os.path.exists(os.path.join(out_dir, "vector.faiss"))
    assert os.path.exists(os.path.join(out_dir, "meta.json"))

    bm25_2, index_2, metas_2 = load_indexes(out_dir)
    assert index_2.ntotal == 3
    assert len(metas_2) == 3
    assert metas_2[0].chunk_id == "c0"
