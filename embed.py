from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Init ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_db", anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="tds_virtual_ta")

# Load processed chunks
with open("processed_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Insert into ChromaDB
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk["text"]).tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk["text"]],
        embeddings=[embedding],
        metadatas=[{
            "source": chunk["source"],
            "title": chunk.get("title", ""),
            "url": chunk.get("url", "")
        }]
    )

print("✅ Done embedding and storing into ChromaDB.")
