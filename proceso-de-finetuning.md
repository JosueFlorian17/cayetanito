pip install unsloth accelerate transformers datasets peft

# 🧠 Fine-tuning de Gemma 3 4B con Unsloth (nutrición infantil Perú 🇵🇪)

Este documento describe todo el proceso para realizar fine-tuning eficiente del modelo `google/gemma-3-4b-pt` usando LoRA, Unsloth y un dataset personalizado con 100 ejemplos de preguntas/respuestas para educación alimentaria infantil con contexto peruano.

---

## ✅ Requisitos previos

- Python 3.10 o superior
- GPU con soporte BF16 o FP16
- Entorno virtual (`venv`) activado

Instalar dependencias:

pip install unsloth accelerate transformers datasets peft

---

## 🗂️ Estructura esperada de archivos

tu_proyecto/
├── nutricion_dataset.json     # Dataset con 100 ejemplos
├── script_finetuning.py       # Script de entrenamiento
└── venv/                      # (opcional) entorno virtual

---

## 📄 Formato del dataset (nutricion_dataset.json)

Debe ser una lista de objetos con campos `prompt` y `response`.

[
  {
    "prompt": "¿Por qué es importante tomar desayuno antes de ir al colegio?",
    "response": "Porque el desayuno te da energía para aprender, jugar y concentrarte mejor en clase."
  },
  ...
]

---

## 🧪 Script completo de entrenamiento

from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch

# 1. Cargar modelo base Gemma 3 4B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-4b-pt",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
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

# 3. Cargar el dataset personalizado
dataset = load_dataset("json", data_files="nutricion_dataset.json", split="train")

# 4. Formatear los ejemplos en estilo instruct
def format_example(example):
    return {
        "text": f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    }

dataset = dataset.map(format_example)

# 5. Preparar el modelo para entrenamiento
FastLanguageModel.for_training(model)

# 6. Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir = "finetuned-gemma-nutricion",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 5,
    learning_rate = 2e-5,
    logging_steps = 5,
    save_steps = 20,
    save_total_limit = 2,
    bf16 = torch.cuda.is_available(),
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

# 8. Ejecutar entrenamiento
trainer.train()

---

## 📦 Salida esperada

Una carpeta `finetuned-gemma-nutricion/` con los siguientes archivos:

- `adapter_model.bin`
- `adapter_config.json`
- `trainer_state.json`

Esta carpeta contiene el fine-tuning LoRA que puedes cargar para hacer inferencia especializada.

---

## 🧠 Siguientes pasos sugeridos

- Usar el modelo fine-tuneado con tu script de inferencia.
- Integrar el sistema de prompt engineering personalizado por aula.
- Conectarlo a un flujo RAG si deseas inyectar contexto actualizado.
- Validar respuestas con niños o docentes reales antes de producción.
