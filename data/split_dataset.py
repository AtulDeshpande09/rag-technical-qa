import json
import random
from datetime import datetime

# ======================
# CONFIG
# ======================
INPUT_FILE = "dataset_semantic_filtered.json"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

LOG_FILE = "preprocessing_log.txt"


# ======================
# LOG FUNCTION
# ======================
def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# ======================
# LOAD DATASET
# ======================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

original_size = len(dataset)
write_log(f"Dataset splitting started. Total samples: {original_size}")
write_log(f"Seed used: {SEED}")

# ======================
# SHUFFLE (reproducible)
# ======================
random.seed(SEED)
random.shuffle(dataset)

# ======================
# SPLIT
# ======================
train_end = int(TRAIN_RATIO * original_size)
val_end = train_end + int(VAL_RATIO * original_size)

train_data = dataset[:train_end]
val_data = dataset[train_end:val_end]
test_data = dataset[val_end:]

# ======================
# SAVE FILES
# ======================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(f"{OUTPUT_DIR}/val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

with open(f"{OUTPUT_DIR}/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

# ======================
# LOG RESULTS
# ======================
write_log(f"Train size: {len(train_data)}")
write_log(f"Validation size: {len(val_data)}")
write_log(f"Test size: {len(test_data)}")
write_log("Dataset splitting completed.\n")

print("Dataset split successfully.")
