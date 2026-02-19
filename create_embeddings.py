import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ======================
# CONFIG
# ======================
DATA_FILE = "data/data/train.jsonl"
DB_PATH = "vector_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DB_PATH, exist_ok=True)

# ======================
# LOAD DATA (JSONL)
# ======================
dataset = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Loaded {len(dataset)} training samples.")

# ======================
# CREATE TEXT CHUNKS
# ======================
texts = []
for item in dataset:
    chunk = f"Question: {item['question']}\nAnswer: {item['answer']}"
    texts.append(chunk)

# ======================
# EMBEDDING MODEL
# ======================
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

# ======================
# FAISS INDEX
# ======================
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity

index.add(embeddings)

print("Saving FAISS index...")
faiss.write_index(index, f"{DB_PATH}/index.faiss")

# ======================
# SAVE TEXT METADATA
# ======================
# Still JSON (not JSONL) because FAISS metadata is easier this way
with open(f"{DB_PATH}/texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, indent=2, ensure_ascii=False)

print("Embeddings created and saved successfully.")
