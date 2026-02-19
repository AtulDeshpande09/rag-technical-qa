import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_DIR = "models/mistral_lora"
MERGED_DIR = "models/mistral_merged"

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to("cuda")

print("Loading LoRA...")

model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("Merging weights...")

model = model.merge_and_unload()

print("Saving merged model...")

model.save_pretrained(MERGED_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Model saved successfully!")
