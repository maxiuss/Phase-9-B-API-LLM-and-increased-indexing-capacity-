import openai
import json
import os
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(chunks, model="text-embedding-ada-002", batch_size=10):
    """
    Call OpenAI API to get embedding vectors for text chunks in batches.
    Returns a list of embedding vectors.
    """
    embeddings = []
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        batch_embeddings = [data["embedding"] for data in response["data"]]
        embeddings.extend(batch_embeddings)
        print(f"Processed batch {(i//batch_size)+1} of {n_batches}")
    return embeddings

def index_text_chunks(chunks, index_path="faiss_index.index", metadata_path="chunk_metadata.json"):
    """
    Create embeddings for each chunk in batches, index them using FAISS,
    and save the index and metadata.
    :param chunks: A list of text chunks (strings).
    :param index_path: File path to save the FAISS index.
    :param metadata_path: File path to save chunk metadata.
    """
    # Generate embeddings for all chunks in batches of 10.
    embeddings = get_embeddings(chunks, batch_size=10)
    
    # Convert embeddings to a numpy array of type float32.
    embedding_matrix = np.array(embeddings).astype('float32')
    
    # Get the dimension from the first embedding.
    dim = embedding_matrix.shape[1]
    
    # Create a FAISS index (using IndexFlatL2 for simplicity).
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)  # add embeddings to index
    
    # Save the index.
    faiss.write_index(index, index_path)
    
    # Save metadata: a mapping of index position to the text chunk.
    metadata = {str(i): chunks[i] for i in range(len(chunks))}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Indexed {len(chunks)} chunks. FAISS index saved to {index_path}, metadata saved to {metadata_path}.")

if __name__ == "__main__":
    # Load your preprocessed text chunks from all_chunks.json
    with open("all_chunks.json", "r") as f:
        text_chunks = json.load(f)
    
    # Index the chunks.
    index_text_chunks(text_chunks)
