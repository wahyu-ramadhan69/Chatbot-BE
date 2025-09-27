import os
import faiss
import pickle
import numpy as np
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-ada-002"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def embed_text(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def embed_documents(doc_folder="rag/documents"):
    docs = []
    vectors = []

    for filename in os.listdir(doc_folder):
        path = os.path.join(doc_folder, filename)

        if filename.endswith(".pdf"):
            content = extract_text_from_pdf(path)
        elif filename.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        if not content.strip():
            continue

        vector = embed_text(content)
        docs.append((filename, content))
        vectors.append(vector)

    if not vectors:
        print("Tidak ada dokumen valid untuk di-embed.")
        return

    # Simpan FAISS index
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    # Simpan data
    with open("rag/doc_store.pkl", "wb") as f:
        pickle.dump(docs, f)

    faiss.write_index(index, "rag/doc_index.faiss")
    print("Selesai menyimpan FAISS dan dokumen.")

if __name__ == "__main__":
    embed_documents()
