from pypdf import PdfReader
import re

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
        word_length = len(word) + 1
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
    sample_pdf = "data/Neil Strauss - The Game_ Penetrating the Secret Society of Pickup Artists (2005, It Books) - libgen.li.pdf"  # Replace with a valid PDF path
    raw_text = parse_pdf(sample_pdf)
    if raw_text:
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text, max_chunk_size=4000)
        print(f"Extracted {len(chunks)} chunks from the document.")
        # Optionally, write chunks to a sample JSON file:
        import json
        with open("sample_chunks.json", "w") as outfile:
            json.dump(chunks, outfile, indent=2)
        print("Chunks saved to sample_chunks.json")
    else:
        print("No text extracted.")