import os
import faiss
import pickle
import numpy as np
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken  # untuk hitung token agar tidak melebihi batas

# Load .env
load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-ada-002"
TOKEN_LIMIT = 8000   # batas aman (8192 token max)
CHUNK_SIZE = 2000    # kira-kira 2000 token per potong
OVERLAP = 200        # potongan tumpang tindih sedikit

# tokenizer bawaan OpenAI
tokenizer = tiktoken.encoding_for_model(EMBED_MODEL)

def extract_text_from_pdf(pdf_path):
    """Baca PDF dan ambil teks dari setiap halaman."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Bagi teks jadi potongan (chunk) kecil biar tidak melebihi token limit."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def embed_text(text):
    """Kirim teks ke model embedding OpenAI."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def embed_documents(doc_folder="rag/documents"):
    """Proses semua dokumen dalam folder, split otomatis, dan simpan FAISS index."""
    docs = []
    vectors = []

    for filename in os.listdir(doc_folder):
        path = os.path.join(doc_folder, filename)
        if filename.startswith("."):
            continue  # skip .DS_Store, dll

        if filename.endswith(".pdf"):
            content = extract_text_from_pdf(path)
        elif filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        if not content.strip():
            continue

        # pecah jadi potongan kecil
        chunks = split_text(content)
        print(f"Embedding {len(chunks)} chunks dari {filename}")

        for idx, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            vectors.append(vector)
            docs.append((f"{filename}_chunk{idx+1}", chunk))

    if not vectors:
        print("Tidak ada dokumen valid untuk di-embed.")
        return

    # Simpan FAISS index
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    # Simpan data teks
    with open("rag/doc_store.pkl", "wb") as f:
        pickle.dump(docs, f)

    faiss.write_index(index, "rag/doc_index.faiss")
    print(f"Selesai menyimpan {len(vectors)} embedding ke FAISS.")

if __name__ == "__main__":
    embed_documents()
