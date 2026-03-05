import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")

# load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# load embeddings and metadata
embeddings = np.load(EMBEDDINGS_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def hybrid_search(question, paper_name, top_k=5):
    # filter chunks by selected paper
    paper_chunks = []
    paper_embeddings = []

    for i, chunk in enumerate(metadata):
        if chunk["paper_name"] == paper_name:
            paper_chunks.append(chunk)
            paper_embeddings.append(embeddings[i])

    if not paper_chunks:
        return "", 0.0

    paper_embeddings = np.array(paper_embeddings)

    # encode question
    question_embedding = model.encode(question)

    # cosine similarity
    scores = util.cos_sim(question_embedding, paper_embeddings)[0]

    # get top results
    top_indices = scores.argsort(descending=True)[:top_k]

    context = []
    # Calculate average confidence of the top_k results
    # Or you can just take the score of the #1 result
    top_scores = [scores[idx].item() for idx in top_indices]
    
    for idx in top_indices:
        context.append(paper_chunks[idx]["text"])

    # Returning both the joined text and the max confidence score
    return " ".join(context), max(top_scores)