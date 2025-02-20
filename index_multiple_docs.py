import os
import json
import argparse
import faiss
import numpy as np
import openai
from dotenv import load_dotenv

from data_ingestion.parse_documents import parse_pdf, clean_text, chunk_text
from semantic_search.semantic_search import load_faiss_index, load_metadata

load_dotenv()  # Ensure .env is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(chunks, model="text-embedding-3-small", batch_size=50):
    """
    Call OpenAI API to get embedding vectors for text chunks in batches.
    Returns a list of embedding vectors.
    """
    embeddings = []
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        batch_embeddings = [data["embedding"] for data in response["data"]]
        embeddings.extend(batch_embeddings)
        print(f"Processed batch {(i // batch_size) + 1} of {n_batches}")
    return embeddings

def index_multiple_docs(pdf_directory="data/pdfs", index_path="faiss_index.index", metadata_path="chunk_metadata.json"):
    """
    Ingest and index multiple PDF files from a directory.
    If an existing FAISS index is found, new embeddings are appended.
    """
    # 1. Gather all PDF files.
    pdf_files = [
        f for f in os.listdir(pdf_directory)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}. Exiting.")
        return

    # 2. Load existing index & metadata if available.
    index = None
    existing_metadata = {}
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = load_faiss_index(index_path)
        existing_metadata = load_metadata(metadata_path)
        print(f"Loaded existing FAISS index with {index.ntotal} vectors.")
    else:
        print("No existing index found. A new one will be created.")

    # 3. Collect all new chunks and metadata.
    all_new_chunks = []
    new_metadata_dict = {}
    offset = 0
    if existing_metadata:
        existing_ids = list(map(int, existing_metadata.keys()))
        offset = max(existing_ids) + 1

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing: {pdf_path}")
        raw_text = parse_pdf(pdf_path)
        if not raw_text:
            print(f"No text extracted from {pdf_file}.")
            continue
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text, max_chunk_size=4000)

        # Use the filename as a document id
        doc_id = pdf_file

        for chunk in chunks:
            all_new_chunks.append(chunk)
            new_metadata_dict[str(offset)] = {
                "doc_id": doc_id,
                "text": chunk
            }
            offset += 1

    if not all_new_chunks:
        print("No new text chunks found. Exiting.")
        return

    # 4. Batch generate embeddings for all collected chunks.
    embeddings = get_embeddings(all_new_chunks, batch_size=50)
    embedding_matrix = np.array(embeddings).astype("float32")

    # 5. Create new index if one doesn't exist, otherwise update.
    if index is None:
        dim = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        print("Created a new FAISS index.")
    index.add(embedding_matrix)
    print(f"Added {len(embeddings)} new vectors to the index.")

    # 6. Save the FAISS index.
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}.")

    # 7. Update and save metadata.
    existing_metadata.update(new_metadata_dict)
    with open(metadata_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}.")

def main():
    parser = argparse.ArgumentParser(description="Index multiple PDF files.")
    parser.add_argument("--pdf_directory", type=str, default="data/pdfs", help="Path to folder with PDFs.")
    parser.add_argument("--index_path", type=str, default="faiss_index.index", help="Path to FAISS index.")
    parser.add_argument("--metadata_path", type=str, default="chunk_metadata.json", help="Path to metadata JSON.")
    args = parser.parse_args()

    index_multiple_docs(
        pdf_directory=args.pdf_directory,
        index_path=args.index_path,
        metadata_path=args.metadata_path
    )

if __name__ == "__main__":
    main()


