import os
import json
import faiss
import time
import random
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def get_embeddings_in_batches(texts, batch_size=20, model="Text-embedding-3-small"):
    """
    Embed a list of texts in batches. Retries on failure with exponential backoff.
    Returns a list of embedding vectors.
    """
    all_embeddings = []
    total = len(texts)
    i = 0
    while i < total:
        batch = texts[i : i + batch_size]
        for _retry in range(5):  # up to 5 retries/py
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=model
                )
                embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                print(f"Embedding error on batch {i}-{i+batch_size}: {e}")
                backoff = 2 ** _retry + random.random()
                time.sleep(backoff)
        else:
            # If we exhausted all retries, skip these items or handle differently
            print(f"Skipping batch {i}-{i+batch_size} after 5 retries.")
        i += batch_size
        processed = min(i, total)
        print(f"Progress: Processed {processed}/{total} chunks.")
    return all_embeddings

def create_or_load_faiss_index(dim, index_path="faiss_index.index"):
    """
    If an index file exists, load it; otherwise, create a new IndexFlatL2.
    """
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Loaded existing index with {index.ntotal} vectors.")
        return index
    else:
        index = faiss.IndexFlatL2(dim)
        print("Created a new FAISS IndexFlatL2.")
        return index

def index_multiple_docs(parsed_json="all_chunks.json", index_path="faiss_index.index", metadata_path="chunk_metadata.json"):
    """
    Reads parsed text chunks from a JSON file, embeds them in batches,
    and adds them to a FAISS index with incremental metadata.
    """
    if not os.path.exists(parsed_json):
        print(f"No parsed data file found at {parsed_json}. Exiting.")
        return

    # Load existing metadata or create new
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        offset = max(map(int, metadata.keys())) + 1 if metadata else 0
    else:
        metadata = {}
        offset = 0

    with open(parsed_json, "r") as f:
        parsed_data = json.load(f)

    all_texts = []
    all_ids = []
    if isinstance(parsed_data, dict):
        for pdf_file, chunks in parsed_data.items():
            print(f"Starting indexing document: {pdf_file} with {len(chunks)} chunks.")
            for chunk in chunks:
                all_texts.append((pdf_file, chunk))
                all_ids.append(str(offset))
                offset += 1
            print(f"Finished indexing document: {pdf_file}")
    elif isinstance(parsed_data, list):
        print(f"Starting indexing of {len(parsed_data)} chunks from unknown document.")
        for chunk in parsed_data:
            all_texts.append(("unknown", chunk))
            all_ids.append(str(offset))
            offset += 1
        print("Finished indexing unknown document.")
    else:
        print("Unexpected format in parsed JSON. Exiting.")
        return

    # Convert texts to a list of just the chunk text
    chunk_texts = [t[1] for t in all_texts]

    # 1. Get embeddings in batches
    embeddings = get_embeddings_in_batches(chunk_texts, batch_size=20, model="text-embedding-3-small")
    if not embeddings:
        print("No embeddings generated. Exiting.")
        return

    # 2. Build or load FAISS index
    dim = len(embeddings[0])  # dimension from the first embedding
    index = create_or_load_faiss_index(dim, index_path=index_path)

    # 3. Add embeddings to FAISS
    embedding_matrix = np.array(embeddings).astype('float32')
    index.add(embedding_matrix)
    faiss.write_index(index, index_path)
    print(f"FAISS index now has {index.ntotal} vectors.")

    # 4. Update metadata
    for idx, (pdf_file, chunk) in zip(all_ids, all_texts):
        metadata[idx] = {
            "doc_id": pdf_file,
            "text": chunk
        }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata updated with {len(all_texts)} new entries.")

if __name__ == "__main__":
    index_multiple_docs()




