from pypdf import PdfReader
import re
import os
import json

def parse_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n\n--- Page {page_num+1} ---\n\n"
                full_text += text
        return full_text.strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def clean_text(text):
    cleaned = re.sub(r"Page \d+ of \d+", "", text)
    cleaned = re.sub(r"-{3,}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def chunk_text(text, max_chunk_size=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_size + word_length <= max_chunk_size:
            current_chunk.append(word)
            current_size += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

if __name__ == "__main__":
    pdf_directory = "data/pdfs"  # folder containing PDFs
    all_chunks = []
    
    # Loop through all files in the pdf directory
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing: {pdf_path}")
            raw_text = parse_pdf(pdf_path)
            if raw_text:
                cleaned_text = clean_text(raw_text)
                chunks = chunk_text(cleaned_text, max_chunk_size=4000)
                print(f"Extracted {len(chunks)} chunks from {filename}")
                all_chunks.extend(chunks)
            else:
                print(f"No text extracted from {filename}.")
    
    print(f"Total extracted chunks: {len(all_chunks)}")
    
    # Optionally, write all chunks to a JSON file for later use
    with open("all_chunks.json", "w") as outfile:
        json.dump(all_chunks, outfile, indent=2)
    print("All chunks saved to all_chunks.json")
