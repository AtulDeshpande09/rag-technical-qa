import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from logger import ExperimentLogger


# ======================
# CONFIG
# ======================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_FILE = "data/data/train.jsonl"
VAL_FILE = "data/data/val.jsonl"

OUTPUT_DIR = "models/mistral_lora"
MERGED_DIR = "models/mistral_merged"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


# ======================
# LOGGER
# ======================
logger = ExperimentLogger("Fine-tuning LoRA")
logger.section("CONFIG")
logger.log(f"Model: {MODEL_NAME}")
logger.log(f"Train file: {TRAIN_FILE}")
logger.log(f"Val file: {VAL_FILE}")


# ======================
# CALLBACK FOR LOSS
# ======================
class LossLoggerCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            for k, v in logs.items():
                self.logger.log(f"Step {step} | {k}: {v}")


# ======================
# LOAD JSONL
# ======================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)

logger.log(f"Loaded {len(train_data)} training samples")
logger.log(f"Loaded {len(val_data)} validation samples")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)


# ======================
# TOKENIZER
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ======================
# PROMPT FORMAT
# ======================
def format_example(example):
    prompt = f"""
You are a technical interview assistant.

Answer the following technical question clearly and concisely.

Question:
{example['question']}

Answer:
{example['answer']}
"""
    return {"text": prompt}


train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)


# ======================
# TOKENIZATION + LABELS
# ======================
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset_train = train_dataset.map(tokenize, batched=True)
tokenized_dataset_val = val_dataset.map(tokenize, batched=True)

logger.log("Tokenization completed")


# ======================
# DATA COLLATOR
# ======================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ======================
# LOAD BASE MODEL
# ======================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)


# ======================
# LORA
# ======================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

logger.section("TRAINABLE PARAMETERS")
model.print_trainable_parameters()


# ======================
# TRAINING
# ======================
training_args = TrainingArguments(
    output_dir="./mistral_qa",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    bf16=True,
    save_strategy="no",
    logging_steps=1,
    save_steps=100,
    report_to="none"
)


# ======================
# TRAINER
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val,
    data_collator=data_collator,
    callbacks=[LossLoggerCallback(logger)]
)


# ======================
# TRAIN
# ======================
logger.section("TRAINING START")
trainer.train()
logger.section("TRAINING COMPLETE")


# ======================
# SAVE LORA
# ======================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.log("LoRA adapter saved.")


# ======================
# MERGE
# ======================
logger.section("MERGING LORA")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

logger.log("Merged model saved.")
logger.section("DONE")
