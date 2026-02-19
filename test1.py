import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================
# CONFIG
# ======================
TEST_FILE = "data/test.json"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILE = "results/vanilla_test.json"

# ======================
# LOAD MODEL
# ======================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# ======================
# LOAD DATA
# ======================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

references = []
questions = []

print("Running vanilla evaluation...")

for item in test_data:
    questions.append(item["question"])
    references.append(item["answer"])


# ======================
# LOGGER
# ======================

from logger import ExperimentLogger

logger = ExperimentLogger("Vanilla LLM")
logger.section("MODEL")
logger.log(f"Model name: {MODEL_NAME}")
logger.log(f"Tokenizer vocab size: {tokenizer.vocab_size}")
logger.log(f"Pad token: {tokenizer.pad_token}")
logger.log(f"EOS token: {tokenizer.eos_token}")


def exact_match(predictions, references):
    matches = 0
    for p, r in zip(predictions, references):
        if p.strip().lower() == r.strip().lower():
            matches += 1
    return matches / len(predictions)



# ======================
# METRICS
# ======================

import evaluate
from bert_score import score

def generate_answers(model, tokenizer, questions, max_tokens=100):
    model.eval()
    outputs = []

    for q in questions:
        prompt = f"""
You are a technical interview assistant.

Answer the following technical question clearly and concisely.

Question:
{q}

Answer:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
		do_sample=False
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Remove prompt from output
        answer = text.split("Answer:")[-1].strip()
        outputs.append(answer)

    return outputs


def compute_and_log_metrics(logger, predictions, references):
    logger.section("AUTOMATIC METRICS")

    # ---------------- BLEU ----------------
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        max_order=4
    )

    logger.log(f"BLEU-4: {bleu_results['bleu']:.4f}")

    # ---------------- ROUGE ----------------
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references
    )

    logger.log(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

    # ---------------- BERTScore ----------------
    # Offload BERTScore to CPU to avoid OutOfMemoryError
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        verbose=False,
        device="cpu" # Added device='cpu'
    )

    logger.log(f"BERTScore F1: {F1.mean().item():.4f}")

    em = exact_match(predictions, references)
    logger.log(f"Exact Match: {em:.4f}")


predictions = generate_answers(model, tokenizer, questions)

# Log sample outputs
logger.section("SAMPLE OUTPUTS")
for q, pred in zip(questions[:5], predictions[:5]):
    logger.log(f"Q: {q}")
    logger.log(f"A: {pred}")
    logger.log("-" * 40)

# Log metrics
compute_and_log_metrics(logger, predictions, references)

os.makedirs("results", exist_ok=True)



results = {
    "predictions": predictions,
    "references": references
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


print("Saved predictions.")
print("Done.")
