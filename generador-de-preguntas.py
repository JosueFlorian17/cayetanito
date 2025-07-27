from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

# Cargar modelo base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Cargar adapter
model = PeftModel.from_pretrained(model, "./")
model.eval()

print("👦 Escribe tu pregunta para Cayetanito (o escribe 'salir' para terminar)\n")
while True:
    pregunta = input("Tú: ")
    if pregunta.strip().lower() in ["salir", "exit", "q"]:
        print("👋 ¡Hasta pronto!")
        break

    prompt = f"<s>[INST] {pregunta.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🧠 Extraer texto luego de [/INST]
    if "[/INST]" in decoded:
        respuesta_raw = decoded.split("[/INST]", 1)[1].strip()
    else:
        respuesta_raw = decoded.strip()

    # 🧼 Cortar antes de cualquier nueva pregunta o instrucción
    stop_tokens = ["<s>[INST]", "\nTú:", "¿Por", "Qué", "cómo", "Cuándo", "Dónde", "Quién", "Cuál"]
    corte = len(respuesta_raw)
    for token in stop_tokens:
        idx = respuesta_raw.find(token)
        if idx != -1:
            corte = min(corte, idx)

    respuesta_final = respuesta_raw[:corte].strip()

    # Limpiar tokens extra
    respuesta_final = respuesta_final.replace("<s>", "").replace("</s>", "").strip()

    print(f"Cayetanito: {respuesta_final}\n")