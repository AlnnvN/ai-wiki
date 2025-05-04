import os
import glob
import numpy as np
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

DOCUMENTS_DIR = "documents"
OUTPUT_DIR = "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_documents(directory):
    docs = []
    files = glob.glob(os.path.join(directory, "*.txt"))
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            docs.append((os.path.basename(filepath), content))
    return docs

def preprocess(text):
    return text.strip()

def split_into_chunks(text, max_sentences=5):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    count = 0
    for sentence in sentences:
        chunk += " " + sentence
        count += 1
        if count >= max_sentences:
            chunks.append(chunk.strip())
            chunk = ""
            count = 0
    if chunk:
        chunks.append(chunk.strip())
    return chunks

documents = load_documents(DOCUMENTS_DIR)
all_chunks = []
chunk_source = []

for filename, text in documents:
    processed_text = preprocess(text)
    chunks = split_into_chunks(processed_text)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_source.append(filename)

print(f"Total de chunks gerados: {len(all_chunks)}")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Salvar o índice FAISS
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss.index"))

# Salvar os dados auxiliares
np.save(os.path.join(OUTPUT_DIR, "chunks.npy"), np.array(all_chunks))
np.save(os.path.join(OUTPUT_DIR, "sources.npy"), np.array(chunk_source))

print("Index e dados auxiliares salvos com sucesso!")
