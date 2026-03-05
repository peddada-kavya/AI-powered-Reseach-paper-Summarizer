import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings...")

embeddings = model.encode(texts, show_progress_bar=True)

print("Saving embeddings...")

np.save(EMBEDDINGS_FILE, embeddings)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4)

print("Embeddings saved successfully")