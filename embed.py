import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient("./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="my_collection", embedding_function=embedding_function)

with open("k8s.txt", "r") as f:
    text = f.read()

collection.add(documents=[text], ids=["k8s_doc"])

print("Document added to ChromaDB collection.")