import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# =====================
# CONFIG
# =====================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "sat_data/sat_lp_train.jsonl"
OUTPUT_DIR = "sat_lora_model"

MAX_SEQ_LEN = 512
EPOCHS = 2
LR = 2e-4

# =====================
# DEVICE (CUDA / CPU)
# =====================
# Adjusted for Windows/CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🖥️ Using device:", device)

# =====================
# LOAD MODEL & TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=512,
    truncation=True
)

# Use float16 for CUDA, float32 for CPU
torch_dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
)

model.to(device)

# =====================
# LORA CONFIG
# =====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =====================
# LOAD DATASET
# =====================
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Data file '{DATA_PATH}' not found!")
    exit(1)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# =====================
# TRAINER
# =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=20,
    save_strategy="epoch",
    report_to=[] # Disable wandb/etc
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

# =====================
# TRAIN
# =====================
print("🚀 Starting training...")
trainer.train()

# =====================
# SAVE MODEL
# =====================
trainer.save_model(OUTPUT_DIR)
print("🎉 SAT LoRA model saved to:", OUTPUT_DIR)
