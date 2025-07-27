from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

# Leer CSV con codificación compatible
df = pd.read_csv("nutricion_conocimiento.csv", encoding="ISO-8859-1")  # o encoding="latin1"
df.columns = df.columns.str.strip()

# Convertir a documentos
docs = []
for _, row in df.iterrows():
    content = row["texto"].strip()
    metadata = {"id": str(row["id"])}
    docs.append(Document(page_content=content, metadata=metadata))

# Cargar modelo de embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Vectorizar y guardar
db = FAISS.from_documents(docs, embedding_model)
db.save_local("vectorstore_nutricion")

print("✅ Vectorstore creado correctamente desde archivo CSV.")