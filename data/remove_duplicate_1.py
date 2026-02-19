import json
from logger import write_log

INPUT_FILE = "dataset.json"
OUTPUT_FILE = "dataset_no_duplicates.json"


def normalize(text):
    return text.strip().lower()


def remove_exact_duplicates(data):
    seen = set()
    filtered = []

    for item in data:
        q = normalize(item["question"])

        if q not in seen:
            seen.add(q)
            filtered.append(item)

    return filtered


# Load dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

original_size = len(dataset)
write_log(f"Exact duplicate removal started. Original size: {original_size}")

filtered_dataset = remove_exact_duplicates(dataset)

filtered_size = len(filtered_dataset)
removed = original_size - filtered_size

write_log(f"Exact duplicates removed: {removed}")
write_log(f"New dataset size: {filtered_size}")

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)

write_log("Exact duplicate removal completed.\n")
