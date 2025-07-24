from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch

# 1. Cargar modelo base Gemma 3 4B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-4b-pt",
    max_seq_length = 2048,
    dtype = torch.float16,       # o "auto"
    load_in_4bit = True,         # usa menos memoria
)

# 2. Activar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
)

# 3. Cargar el dataset
dataset = load_dataset("json", data_files="nutricion_dataset.json", split="train")

# 4. Formatear datos
def format_example(example):
    return {
        "text": f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    }

dataset = dataset.map(format_example)

# 5. Activar modo entrenamiento
FastLanguageModel.for_training(model)

# 6. Configurar entrenamiento
training_args = TrainingArguments(
    output_dir = "finetuned-gemma-nutricion",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 5,
    learning_rate = 2e-5,
    logging_steps = 5,
    save_steps = 20,
    save_total_limit = 2,
    bf16 = torch.cuda.is_available(),  # usa bf16 si tu GPU lo soporta
    fp16 = not torch.cuda.is_bf16_supported(),
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit"
)

# 7. Entrenador
trainer = Trainer(
    model = model,
    train_dataset = dataset,
    args = training_args,
    tokenizer = tokenizer,
)

# 8. ¡Entrenar!
trainer.train()
