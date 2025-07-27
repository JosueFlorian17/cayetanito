import json
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# === Cargar configuración del aula ===
with open("salon-data.json", "r", encoding="utf-8") as f:
    config_aula = json.load(f)

# === Construir prompt contextual ===
def construir_prompt(pregunta, contexto):
    prompt = f"""
<s>[INST]
Eres un asistente de nutrición infantil llamado Cayetanito. Estás ayudando al aula "{config_aula['aula']}".
Palabra clave de atención: "{config_aula['palabra_clave_asamblea']}".

Los niños han aprendido sobre: {', '.join(config_aula['temas_aprendidos'])}.
Tienen preferencias locales como: {', '.join(config_aula['preferencias_locales'])}.

Usa un lenguaje simple y amigable para responder.

Pregunta: {pregunta}
Contexto adicional: {contexto}
[/INST]"""
    return prompt.strip()

# === Cargar modelo fine-tuned ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "./")
model.eval()

# === Cargar vectorstore RAG ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("vectorstore_nutricion", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# === Inferencia interactiva ===
print("\n👦 Escribe tu pregunta para Cayetanito (o escribe 'salir' para terminar)\n")
while True:
    pregunta = input("Tú: ").strip()
    if pregunta.lower() in ["salir", "exit", "q"]:
        print("👋 ¡Hasta pronto!")
        break

    resultados = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in resultados[:3]])

    prompt = construir_prompt(pregunta, contexto)
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
    respuesta_raw = decoded.split("[/INST]", 1)[-1].strip()

    stop_tokens = ["<s>[INST]", "\nTú:", "¿Por", "Qué", "cómo", "Cuándo", "Dónde", "Quién", "Cuál"]
    corte = len(respuesta_raw)
    for token in stop_tokens:
        idx = respuesta_raw.find(token)
        if idx != -1:
            corte = min(corte, idx)

    respuesta_final = respuesta_raw[:corte].replace("<s>", "").replace("</s>", "").strip()
    print(f"Cayetanito: {respuesta_final}\n")