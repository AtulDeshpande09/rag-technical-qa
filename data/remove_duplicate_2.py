import json
import numpy as np
from sentence_transformers import SentenceTransformer
from logger import write_log

INPUT_FILE = "dataset_no_duplicates.json"
OUTPUT_FILE = "dataset_semantic_filtered.json"

SIM_THRESHOLD = 0.90

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_filter(dataset):
    embeddings = []
    filtered = []

    for item in dataset:
        question = item["question"]
        emb = model.encode(question, normalize_embeddings=True)

        keep = True
        for prev_emb in embeddings:
            sim = cosine_similarity(emb, prev_emb)
            if sim > SIM_THRESHOLD:
                keep = False
                break

        if keep:
            embeddings.append(emb)
            filtered.append(item)

    return filtered


# Load dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

original_size = len(dataset)

write_log("Semantic filtering started.")
write_log(f"Original dataset size: {original_size}")
write_log(f"Similarity threshold: {SIM_THRESHOLD}")

filtered_dataset = semantic_filter(dataset)

filtered_size = len(filtered_dataset)
removed = original_size - filtered_size

write_log(f"Semantic duplicates removed: {removed}")
write_log(f"Final dataset size: {filtered_size}")

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)

write_log("Semantic filtering completed.\n")
