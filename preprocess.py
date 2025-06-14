import json
import os
def load_data(file_path):
    """Load data from a .jsonl file (JSON Lines format)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def chunk_text(text, max_words=300):
    """Split text into chunks of up to `max_words`."""
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def preprocess_discourse(data):
    """Preprocess Dicourse content: chunk and label each entry."""
    output = []
    for item in data:
        content = item.get("content", "")
        if not content.strip():
            continue
        chunks = chunk_text(content)
        for chunk in chunks:
            output.append({
                "text": chunk,
                "source": "discourse",
                "title": item.get("title", ""),
                "url": item.get("url", "")
            })
    return output

def preprocess_course(data):
    """Preprocess Course content: chunk and label each entry."""
    output = []
    for item in data:
        content = item.get("content", "")
        if not content.strip():
            continue
        chunks = chunk_text(content)
        for chunk in chunks:
            output.append({
                "text": chunk,
                "source": "course",
                "title": item.get("title", "")
            })
    return output

def main():
    discourse_file = "DicourseData.jsonl"
    course_file = "CourseContentData.jsonl"

    try:
        discourse_data = load_data(discourse_file)
        course_data = load_data(course_file)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    processed = preprocess_discourse(discourse_data) + preprocess_course(course_data)

    output_file = "processed_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅ Preprocessing complete. Total chunks written: {len(processed)}")
    print(f"📄 Output saved to: {output_file}")

if __name__ == "__main__":
    main()
