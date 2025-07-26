from unsloth import FastLanguageModel
from transformers import TextStreamer
from peft import PeftModel

# 1. Cargar modelo base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-4b-it",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Cargar el adapter LoRA desde la ruta local
model = PeftModel.from_pretrained(model, "./")
model.eval()

# 3. Streamer para mostrar texto mientras se genera
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 4. Bucle interactivo
print("👦 Escribe tu pregunta para Cayetanito (o escribe 'salir' para terminar)\n")
while True:
    pregunta = input("Tú: ")
    if pregunta.strip().lower() in ["salir", "exit", "q"]:
        print("👋 ¡Hasta pronto!")
        break

    prompt = f"<s>[INST] {pregunta} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    print()  # salto de línea tras la respuesta
