import json
import requests
import time

def get_embedding(text):
    try:
        res = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        res.raise_for_status()
        return res.json().get("embedding")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    input_file = "processed_chunks.json"
    output_file = "embedded_chunks.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Could not load {input_file}: {e}")
        return

    embedded_data = []

    for i, chunk in enumerate(data):
        print(f"🔄 Embedding chunk {i+1}/{len(data)}")

        embedding = get_embedding(chunk["text"])
        if embedding is None:
            print(f"⚠️ Skipping chunk {i} due to missing embedding.")
            continue

        embedded_data.append({
            "text": chunk["text"],
            "embedding": embedding,
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "url": chunk.get("url", "")
        })

        time.sleep(0.2)  # optional: avoid overloading Ollama

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(embedded_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Done! Saved {len(embedded_data)} chunks to {output_file}")
    except Exception as e:
        print(f"❌ Could not save file: {e}")

if __name__ == "__main__":
    main()
