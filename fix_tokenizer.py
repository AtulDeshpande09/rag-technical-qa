from transformers import AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MERGED_DIR = "models/mistral_merged"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

tokenizer.save_pretrained(MERGED_DIR)

print("Tokenizer fixed.")
