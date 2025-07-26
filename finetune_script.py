from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import json

# 1. Configuración de quantización y LoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 2. Cargar modelo base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 3. Aplicar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# 4. Cargar dataset
dataset = load_dataset("json", data_files="nutricion_dataset.json")["train"]

# 5. Preprocesar y tokenizar
def format(example):
    prompt = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    tokens = tokenizer.tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format, remove_columns=["prompt", "response"])

# 6. Data collator para entrenamiento autoregresivo
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer.tokenizer,
    mlm=False,
)

# 7. Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="gemma3-nutricion-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    bf16=True,
    optim="adamw_8bit",
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

# 8. Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer.tokenizer,  # evitar warning de `tokenizer` deprecated
    train_dataset=dataset,
    args=training_args,
    data_collator=collator,
)

# 9. Entrenar
model.config.use_cache = False
trainer.train()

# 10. Guardar modelo y tokenizer
model.save_pretrained("gemma3-nutricion-finetuned")
tokenizer.tokenizer.save_pretrained("gemma3-nutricion-finetuned")
