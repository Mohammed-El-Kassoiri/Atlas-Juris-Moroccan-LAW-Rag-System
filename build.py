#!/usr/bin/env python3
# build.py — Build FAISS index only
# Usage:
#   python build.py --build

import json, pickle, argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

# -------- CONFIG --------
FLATTENED_JSONL = r"data\processed\flattened_docs.jsonl"
INDEX_FILE = "law_index.faiss"
META_FILE = "law_meta.pkl"
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# ------------------------


def load_flattened_docs(path=FLATTENED_JSONL):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found.")
    docs = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "source" not in obj or not obj["source"]:
                obj["source"] = obj.get("bab") or obj.get("fasl") or ""
            docs.append(obj)
    print(f"[i] Loaded {len(docs)} docs.")
    return docs


def build_index(docs):
    print("[i] Loading embedder:", EMB_MODEL_NAME)
    embedder = SentenceTransformer(EMB_MODEL_NAME)

    texts = [d.get("text", "") for d in docs]
    print("[i] Computing embeddings...")
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    faiss.normalize_L2(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(docs, f)

    print(f"[i] Saved index → {INDEX_FILE}")
    print(f"[i] Saved metadata → {META_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build FAISS index")
    args = parser.parse_args()

    if not args.build:
        print("Usage: python RAG.py --build")
        exit()

    docs = load_flattened_docs()
    build_index(docs)
    print("[i] Done.")