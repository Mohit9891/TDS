import requests

payload = {
    "model": "llama3",  # or any model name you're running
    "prompt": "This is a test sentence for embeddings"
}

try:
    r = requests.post("http://localhost:11434/api/embeddings", json=payload)
    print("✅ Success:", r.json())
except Exception as e:
    print("❌ Failed:", e)
