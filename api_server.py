from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
import requests

app = Flask(__name__)

# Load embedded data
with open("embedded_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
metadatas = [item for item in data]
embeddings = np.array([item["embedding"] for item in data]).astype("float32")

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def get_query_embedding(query):
    """Get embedding for the question using your local LLM API (e.g., Ollama)."""
    try:
        res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": query})
        res.raise_for_status()
        return np.array(res.json()["embedding"], dtype="float32")
    except Exception as e:
        print("❌ Embedding failed:", e)
        return None

def generate_answer(context, question):
    """Generate answer using your local LLM."""
    prompt = f"""
You are a Teaching Assistant for the TDS course. Based on the following context, answer the student's question.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",       # make sure this matches the pulled model name
                "prompt": prompt,
                "stream": False,            # ensures whole output is returned
                "raw": False                # make sure it returns plain text, not tokens
            }
        )
        json_data = res.json()
        print("🧠 LLM raw response:", json_data)

        return json_data.get("response", "").strip()

    except Exception as e:
        print("❌ Generation failed:", e)
        return "Sorry, I couldn't generate a response at the moment."

    except Exception as e:
        print("❌ Generation failed:", e)
        return "Sorry, I couldn't generate a response at the moment."



@app.route("/api/", methods=["GET", "POST"])
def handle_question():
    if request.method == "GET":
        return jsonify({"message": "API is working. Use POST with a question."})

    req = request.get_json()
    question = req.get("question", "")
    # ... rest of your POST code ...

    if not question:
        return jsonify({"error": "Missing question"}), 400

    q_embed = get_query_embedding(question)
    if q_embed is None:
        return jsonify({"error": "Failed to get embedding"}), 500

    # Search top 5 chunks
    D, I = index.search(np.array([q_embed]), k=5)
    context_chunks = [texts[i] for i in I[0]]

    # Collect links from top matches
    links = []
    for i in I[0]:
        meta = metadatas[i]
        if meta.get("source") == "discourse" and meta.get("url"):
            links.append({
                "url": meta["url"],
                "text": meta.get("title") or meta.get("text", "")[:60]

            })

    context = "\n\n".join(context_chunks)
    answer = generate_answer(context, question)

    return jsonify({
        "answer": answer,
        "links": links
    })

if __name__ == "__main__":
    app.run(port=8000, debug=True)
