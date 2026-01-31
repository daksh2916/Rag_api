from fastapi import FastAPI
from chromadb.utils import embedding_functions
import chromadb
import ollama

app = FastAPI()
client = chromadb.PersistentClient("./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="my_collection", embedding_function=embedding_function)

@app.post("/query")
def query(query: str):
    results = collection.query(query_texts=[query], n_results=3)
    documents = results['documents'][0]

    context = documents[0] if documents else "No relevant documents found."

    answer = ollama.generate(
        model = "tinyllama",
        prompt=f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    )

    return {"answer": answer['response']}