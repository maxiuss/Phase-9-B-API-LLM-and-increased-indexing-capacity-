import os
import re
import json
import concurrent.futures
from pypdf import PdfReader

def parse_pdf(file_path):
    """
    Extract text from a single PDF file.
    """
    try:
        reader = PdfReader(file_path)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        return "\n".join(full_text).strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def clean_text(text):
    """
    Enhanced cleaning function. Customize regex to match your SOP formatting.
    """
    text = re.sub(r"Page \d+(\s+of\s+\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, max_chunk_size=4000):
    """
    Split text into chunks of up to max_chunk_size characters.
    Adjust as needed based on your LLM's token limit.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        length = len(word) + 1  # +1 for space
        if current_size + length <= max_chunk_size:
            current_chunk.append(word)
            current_size += length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def parse_and_chunk_pdf(file_path, max_chunk_size=4000):
    """
    Full pipeline for a single PDF: parse, clean, chunk.
    Returns a list of text chunks.
    """
    raw_text = parse_pdf(file_path)
    cleaned = clean_text(raw_text)
    return chunk_text(cleaned, max_chunk_size=max_chunk_size)

def process_pdfs_in_parallel(pdf_dir="data/pdfs", max_workers=4):
    """
    Parse and chunk all PDFs in parallel.
    Returns a dict: { "pdf_filename": [chunk1, chunk2, ...], ... }
    """
    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]
    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(parse_and_chunk_pdf, pdf_path): pdf_path
            for pdf_path in pdf_files
        }
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                chunks = future.result()
                results[os.path.basename(pdf_path)] = chunks
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    return results

if __name__ == "__main__":
    pdf_dir = "data/pdfs"
    parsed_data = process_pdfs_in_parallel(pdf_dir, max_workers=8)
    # Write results to a JSON file for inspection or further processing
    with open("all_chunks.json", "w") as f:
        json.dump(parsed_data, f, indent=2)
    print(f"Parsed data for {len(parsed_data)} PDFs.")

