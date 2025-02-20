# local_qa_openai.py

import os
import openai
from dotenv import load_dotenv
from semantic_search.semantic_search import semantic_search

load_dotenv()

# Load OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")

# Set the API key for openai
openai.api_key = OPENAI_API_KEY

def answer_question_openai(question, k=3, model="gpt-3.5-turbo"):
    """
    Retrieves context using cached FAISS index/metadata, builds a prompt,
    and queries OpenAI for an answer.

    :param question: The user's question (string).
    :param k: Number of context chunks to retrieve.
    :param model: The OpenAI model name (default: "gpt-3.5-turbo").
    :return: The generated answer (string).
    """
    # 1. Retrieve relevant chunks from FAISS
    context_chunks = semantic_search(question, k=k)
    # Extract 'text' from each metadata dictionary
    context = "\n".join(
        chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk)
        for chunk in context_chunks
    )

    # 2. Build the system + user messages
    # Example: Provide context as a system message, and the question as a user message
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question:\n{context}"},
        {"role": "user", "content": question}
    ]

    # 3. Call OpenAI's ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
    except Exception as e:
        return f"OpenAI API request failed: {e}"

    # 4. Extract the assistant's reply
    try:
        answer = response.choices[0].message["content"].strip()
        return answer
    except (KeyError, IndexError):
        return "No valid answer returned from OpenAI."

if __name__ == "__main__":
    # Simple CLI test
    question = input("Enter your question: ")
    answer = answer_question_openai(question)
    print("\nAnswer from OpenAI:")
    print(answer)

