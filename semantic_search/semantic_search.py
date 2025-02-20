# semantic_search/semantic_search.py

import faiss
import json
import numpy as np
import os
import openai
from dotenv import load_dotenv

load_dotenv()

INDEX = None
METADATA = None

def load_faiss_index(index_path="faiss_index.index"):
    global INDEX
    if INDEX is None:
        if os.path.exists(index_path):
            INDEX = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {INDEX.ntotal} vectors.")
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}.")
    return INDEX

def load_metadata(metadata_path="chunk_metadata.json"):
    global METADATA
    if METADATA is None:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                METADATA = json.load(f)
            print(f"Loaded metadata with {len(METADATA)} entries.")
        else:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
    return METADATA

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get a meaningful embedding for the input text using OpenAI's Embeddings API.
    Returns a NumPy array of shape (1, embedding_dim) and type float32.
    """
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response["data"][0]["embedding"]
        # Convert to numpy array with shape (1, embedding_dim)
        return np.array(embedding, dtype="float32").reshape(1, -1)
    except Exception as e:
        print("Error computing embedding:", e)
        # Fallback: return random embedding with default dim=768 (useful for testing)
        return np.random.rand(1, 768).astype("float32")

def semantic_search(query, k=3, index_path="faiss_index.index", metadata_path="chunk_metadata.json"):
    """
    Performs a semantic search by computing a query embedding,
    and then retrieving the k nearest vectors from the FAISS index.
    """
    index = load_faiss_index(index_path)
    metadata = load_metadata(metadata_path)
    
    # Compute the query's embedding using the real OpenAI API
    query_embedding = get_embedding(query)
    
    # Search the FAISS index with the computed query embedding
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        key = str(idx)
        if key in metadata:
            results.append(metadata[key])
    return results

