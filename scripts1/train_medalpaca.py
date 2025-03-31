# train_medalpaca.py

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig

# Step 1: Load base model & tokenizer
base_model = "PrunaAI/medalpaca-medalpaca-7b-bnb-4bit-smashed"
tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b", trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load the dataset
data = load_dataset("json", data_files=r"C:\Users\advai\OneDrive\Desktop\gen\medical_bot\data\jsonl\drug_data_full.jsonl", split="train")

# Step 3: Format the dataset
def format_prompt(example):
    prompt = f"### User: {example['prompt']}\n### Assistant: {example['response']}"
    return {"text": prompt}

data = data.map(format_prompt)

# Step 4: Tokenize the dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_data = data.map(tokenize, batched=True)
tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Step 5: Load the model with 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    #quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Step 6: Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# Step 7: Define TrainingArguments
training_args = TrainingArguments(
    output_dir=r"C:\Users\advai\OneDrive\Desktop\gen\medical_bot\scripts",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    evaluation_strategy="no",
    learning_rate=2e-5,
    report_to="none"
)

# Step 8: Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 9: Start Training
trainer.train()
model.save_pretrained(r"C:\Users\advai\OneDrive\Desktop\gen\medical_bot\scripts")
tokenizer.save_pretrained(r"C:\Users\advai\OneDrive\Desktop\gen\medical_bot\scripts")
