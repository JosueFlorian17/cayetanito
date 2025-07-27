from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === RAG SETUP ===
loader = CSVLoader(file_path="nutricion_conocimiento.csv", source_column="texto")
docs = loader.load()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.from_documents(docs, embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === MODELO LOCAL ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "./")
model.eval()

print("👦 Escribe tu pregunta para Cayetanito (o escribe 'salir' para terminar)\n")
while True:
    pregunta = input("Tú: ").strip()
    if pregunta.lower() in ["salir", "exit", "q"]:
        print("👋 ¡Hasta pronto!")
        break

    # === Recuperar contexto con LangChain ===
    resultados = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in resultados])

    # === Construir prompt enriquecido ===
    prompt = f"<s>[INST] Usa la siguiente información para ayudarme:\n{contexto}\n\nPregunta: {pregunta} [/INST]"

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
    respuesta = decoded.split("[/INST]", 1)[1].strip() if "[/INST]" in decoded else decoded.strip()

    # Limpiar
    stop_tokens = ["<s>[INST]", "\nTú:", "¿Por", "Qué", "cómo", "Cuándo", "Dónde", "Quién", "Cuál"]
    corte = min([respuesta.find(t) if respuesta.find(t) != -1 else len(respuesta) for t in stop_tokens])
    print(f"Cayetanito: {respuesta[:corte].replace('<s>', '').replace('</s>', '').strip()}\n")
