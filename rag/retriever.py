import faiss
import pickle
import numpy as np
from .embeddings import embed_text

# Load index dan dokumen
index = faiss.read_index("rag/doc_index.faiss")
with open("rag/doc_store.pkl", "rb") as f:
    documents = pickle.load(f)

def retrieve_docs(query, top_k=2):
    vec = np.array([embed_text(query)]).astype("float32")
    scores, indices = index.search(vec, top_k)

    results = []
    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i][1])  # ambil isi dokumen
    return results
