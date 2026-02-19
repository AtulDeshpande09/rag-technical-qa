import json
import os
import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ======================
# CONFIG
# ======================
TEST_FILE = "data/test.json"

FINETUNED_MODEL = "models/mistral_merged"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_DB = "vector_db"
TOP_K = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "results/rag_finetuned_test.json"

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================
# LOAD MODEL
# ======================
print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# ======================
# LOAD RETRIEVER
# ======================
print("Loading retriever...")

embed_model = SentenceTransformer(EMBED_MODEL)

index = faiss.read_index(f"{VECTOR_DB}/index.faiss")

with open(f"{VECTOR_DB}/texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# ======================
# LOAD DATA
# ======================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

questions = []
references = []

for item in test_data:
    questions.append(item["question"])
    references.append(item["answer"])

# ======================
# LOGGER
# ======================
from logger import ExperimentLogger

logger = ExperimentLogger("RAG + Fine-tuned")
logger.section("MODEL")
logger.log(f"Fine-tuned model: {FINETUNED_MODEL}")
logger.log(f"Retriever: {EMBED_MODEL}")
logger.log(f"Top-K: {TOP_K}")

# ======================
# RETRIEVAL
# ======================
def retrieve_context(question):
    q_emb = embed_model.encode(
        question,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(np.array([q_emb]), TOP_K)
    retrieved = [texts[i] for i in indices[0]]

    return retrieved

# ======================
# GENERATION
# ======================
def generate_answers(model, tokenizer, questions):
    model.eval()
    outputs = []

    for q in tqdm(questions):
        context_list = retrieve_context(q)
        context = "\n\n".join(context_list)

        prompt = f"""
You are a technical interview assistant.

Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{q}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
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
# RUN
# ======================
print("Running RAG + Fine-tuned evaluation...")

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
import evaluate
from bert_score import score

def exact_match(predictions, references):
    matches = 0
    for p, r in zip(predictions, references):
        if p.strip().lower() == r.strip().lower():
            matches += 1
    return matches / len(predictions)

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

compute_and_log_metrics(logger, predictions, references)

# ======================
# SAVE
# ======================
os.makedirs("results", exist_ok=True)

results = {
    "predictions": predictions,
    "references": references
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved predictions.")
print("Done.")
