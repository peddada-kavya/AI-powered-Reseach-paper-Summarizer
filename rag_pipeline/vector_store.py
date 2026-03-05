import json
import os
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_FILE = os.path.join(BASE_DIR, "rag_pipeline", "chunks.json")

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, convert_to_tensor=True)


def search(question, paper_name, top_k=3, threshold=0.55):

    q_emb = model.encode(question, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, embeddings)[0]

    results = []

    for i, score in enumerate(scores):

        if chunks[i]["paper_name"] == paper_name and score >= threshold:

            results.append({
                "text": chunks[i]["text"],
                "score": float(score)
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]