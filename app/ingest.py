import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path: str, source_name: str):
    print(f"Ingesting {pdf_path}...")

    # Extract text from PDF
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + " "

    full_text = clean_text(full_text)
    chunks = chunk_text(full_text)
    print(f"  Created {len(chunks)} chunks")

    # Embed and upsert to Pinecone
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = embedding_model.encode(batch).tolist()
        vectors = [
            {
                "id": f"{source_name}-chunk-{i+j}",
                "values": embeddings[j],
                "metadata": {"text": batch[j], "source": source_name}
            }
            for j in range(len(batch))
        ]
        index.upsert(vectors=vectors)
        print(f"  Upserted batch {i//batch_size + 1}")

    print(f"Done ingesting {source_name}!")

if __name__ == "__main__":
    pdf_path = sys.argv[1]
    source_name = sys.argv[2]
    ingest_pdf(pdf_path, source_name)