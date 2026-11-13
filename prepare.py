import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = "knowledge-base"   # путь к базе знаний
CHUNK_SIZE = 500            # слов в чанке
OVERLAP = 75                # перекрытие

def read_doc(path, f):
    """Читает файл и извлекает метаданные из имени: категория__название.txt"""
    rel = os.path.relpath(path, KB_DIR)
    parts = f.replace(".txt", "").split("__", 1)
    category = parts[0] if len(parts) > 1 else "прочее"
    title = parts[1] if len(parts) > 1 else parts[0]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return {"doc": title, "category": category, "path": rel, "text": text}

def split_text(text, meta):
    """Режет текст документа на чанки, добавляя метаданные к каждому."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        piece = " ".join(words[i:i + CHUNK_SIZE])
        if len(piece) > 100:  # отбрасываем слишком короткие куски
            chunks.append({**meta, "text": piece})
    return chunks

# 1. Собираем все документы
docs = []
for root, _, files in os.walk(KB_DIR):
    for f in sorted(files):
        if f.endswith(".txt"):
            docs.append(read_doc(os.path.join(root, f), f))

print(f"Документов: {len(docs)}")

# 2. Режем на чанки
chunks = []
for d in docs:
    chunks.extend(split_text(d["text"], {k: v for k, v in d.items() if k != "text"}))

print(f"Чанков: {len(chunks)}")

# 3. Эмбеддинги + индекс
model = SentenceTransformer("BAAI/bge-m3")
vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(np.array(vectors, dtype="float32"))

# 4. Сохраняем
faiss.write_index(index, "index.faiss")
with open("chunks.json", "w", encoding="utf-8") as fh:
    json.dump(chunks, fh, ensure_ascii=False)
print("Готово: index.faiss + chunks.json")