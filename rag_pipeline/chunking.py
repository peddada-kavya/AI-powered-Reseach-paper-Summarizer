import os
import json
import uuid
from pathlib import Path
import fitz  # PyMuPDF

# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_FOLDER = BASE_DIR / "pdf_files"
OUTPUT_FILE = BASE_DIR / "rag_pipeline" / "chunks.json"

# =========================
# CHUNK SETTINGS (OPTIMIZED)
# =========================

CHUNK_SIZE = 150      # words per chunk
OVERLAP = 30          # overlap words


# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_text_from_pdf(pdf_path):
    """Extract full text from PDF"""

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text.strip()


# =========================
# TEXT CHUNKING
# =========================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks"""

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if len(chunk.strip()) > 30:  # ignore tiny chunks
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================
# PROCESS ALL PDFs
# =========================

def process_all_pdfs():

    if not PDF_FOLDER.exists():
        print("pdf_files folder not found")
        return

    all_chunks = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found")
        return

    print(f"Found {len(pdf_files)} PDFs")

    for paper_index, pdf_file in enumerate(pdf_files):

        paper_id = f"P{paper_index}"

        print(f"Processing: {pdf_file.name}")

        text = extract_text_from_pdf(pdf_file)

        if not text:
            print("No text extracted")
            continue

        chunks = chunk_text(text)

        print(f"Created {len(chunks)} chunks")

        for chunk_index, chunk in enumerate(chunks):

            chunk_data = {

                "paper_id": paper_id,

                "paper_name": pdf_file.name,

                "chunk_id": f"{paper_id}_C{chunk_index}",

                "text": chunk

            }

            all_chunks.append(chunk_data)

    # Save chunks
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print("\nChunking complete")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    process_all_pdfs()