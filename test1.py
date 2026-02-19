import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ======================
# CONFIG
# ======================
TEST_FILE = "data/test.jsonl"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILE = "results/vanilla_test.jsonl"

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
# LOAD DATA (JSONL)
# ======================
questions = []
references = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        questions.append(item["question"])
        references.append(item["answer"])

print(f"Loaded {len(questions)} test samples.")
print("Running vanilla evaluation...")

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


# ======================
# EXACT MATCH
# ======================
def exact_match(predictions, references):
    matches = 0
    for p, r in zip(predictions, references):
        if p.strip().lower() == r.strip().lower():
            matches += 1
    return matches / len(predictions)


# ======================
# GENERATION
# ======================
def generate_answers(model, tokenizer, questions, max_tokens=100):
    model.eval()
    outputs = []

    for q in tqdm(questions):
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
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = text.split("Answer:")[-1].strip()
        outputs.append(answer)

    return outputs


# ======================
# METRICS
# ======================
import evaluate
from bert_score import score


def compute_and_log_metrics(logger, predictions, references):
    logger.section("AUTOMATIC METRICS")

    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        max_order=4
    )
    logger.log(f"BLEU-4: {bleu_results['bleu']:.4f}")

    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references
    )
    logger.log(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        verbose=False,
        device="cpu"
    )
    logger.log(f"BERTScore F1: {F1.mean().item():.4f}")

    em = exact_match(predictions, references)
    logger.log(f"Exact Match: {em:.4f}")


# ======================
# RUN
# ======================
predictions = generate_answers(model, tokenizer, questions)

# ======================
# SAMPLE OUTPUTS
# ======================
logger.section("SAMPLE OUTPUTS")

for q, pred in zip(questions[:5], predictions[:5]):
    logger.log(f"Q: {q}")
    logger.log(f"A: {pred}")
    logger.log("-" * 40)

# ======================
# METRICS
# ======================
compute_and_log_metrics(logger, predictions, references)

# ======================
# SAVE JSONL RESULTS
# ======================
os.makedirs("results", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for q, pred, ref in zip(questions, predictions, references):
        f.write(json.dumps({
            "question": q,
            "prediction": pred,
            "reference": ref
        }, ensure_ascii=False) + "\n")

print("Saved predictions.")
print("Done.")
