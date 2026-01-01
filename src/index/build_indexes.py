from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import orjson
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # we will raise a clear error at runtime if used


CHUNKS_PATH_DEFAULT = "data/processed/chunks.jsonl"
INDEX_DIR_DEFAULT = "indexes"

BM25_PATH = "bm25.pkl"
FAISS_PATH = "vector.faiss"
META_PATH = "meta.json"


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    # deterministic, simple tokenization
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: str
    doc_id: str
    header_path: str
    source_type: str


def load_chunks_jsonl(path: str) -> Tuple[List[str], List[ChunkMeta]]:
    texts: List[str] = []
    metas: List[ChunkMeta] = []

    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            texts.append(obj["text"])
            metas.append(
                ChunkMeta(
                    chunk_id=obj["chunk_id"],
                    doc_id=obj.get("doc_id", ""),
                    header_path=obj.get("header_path", "") or "",
                    source_type=obj.get("source_type", "") or "",
                )
            )

    if not texts:
        raise ValueError(f"No chunks found in {path}")
    return texts, metas


def build_bm25(texts: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def build_faiss_index(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Tuple["faiss.Index", np.ndarray]:
    if faiss is None:
        raise RuntimeError(
            "faiss is not available. If you're on macOS and faiss-cpu fails, "
            "we can swap to hnswlib. Paste the import error and I’ll adjust."
        )

    model = SentenceTransformer(model_name, device="cpu")

    # Encode + normalize => cosine similarity with inner product
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


def save_indexes(
    out_dir: str,
    bm25: BM25Okapi,
    faiss_index: "faiss.Index",
    metas: List[ChunkMeta],
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # BM25 (pickle)
    with open(os.path.join(out_dir, BM25_PATH), "wb") as f:
        pickle.dump(bm25, f)

    # FAISS
    faiss.write_index(faiss_index, os.path.join(out_dir, FAISS_PATH))

    # Metadata
    meta_obj = [
        {
            "chunk_id": m.chunk_id,
            "doc_id": m.doc_id,
            "header_path": m.header_path,
            "source_type": m.source_type,
        }
        for m in metas
    ]
    with open(os.path.join(out_dir, META_PATH), "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)


def load_indexes(in_dir: str) -> Tuple[BM25Okapi, "faiss.Index", List[ChunkMeta]]:
    with open(os.path.join(in_dir, BM25_PATH), "rb") as f:
        bm25: BM25Okapi = pickle.load(f)

    if faiss is None:
        raise RuntimeError("faiss not available; cannot load vector index.")

    faiss_index = faiss.read_index(os.path.join(in_dir, FAISS_PATH))

    with open(os.path.join(in_dir, META_PATH), "r", encoding="utf-8") as f:
        raw = json.load(f)
    metas = [ChunkMeta(**x) for x in raw]
    return bm25, faiss_index, metas


def main(
    chunks_path: str = CHUNKS_PATH_DEFAULT,
    out_dir: str = INDEX_DIR_DEFAULT,
) -> None:
    texts, metas = load_chunks_jsonl(chunks_path)

    bm25, _ = build_bm25(texts)
    vindex, _ = build_faiss_index(texts)

    save_indexes(out_dir, bm25, vindex, metas)
    print(f"Saved BM25 + FAISS + metadata to: {out_dir}/")


if __name__ == "__main__":
    main()