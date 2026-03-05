import os
import json
from PyPDF2 import PdfReader


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct absolute paths
pdf_folder = os.path.join(BASE_DIR, "pdf_files")
abstract_folder = os.path.join(BASE_DIR, "abstracts")
output_file = os.path.join(BASE_DIR, "ingested_data.json")


def ingest_pdf():

    data = []

    # Create folders if not exist
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(abstract_folder, exist_ok=True)

    print("Reading PDFs from:", pdf_folder)

    for file in os.listdir(pdf_folder):

        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(pdf_folder, file)

            print("Processing:", file)

            # Extract PDF text
            reader = PdfReader(pdf_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    full_text += text


            # Read abstract
            abstract_file = file.replace(".pdf", ".txt")
            abstract_path = os.path.join(abstract_folder, abstract_file)

            abstract_text = ""

            if os.path.exists(abstract_path):

                with open(abstract_path, "r", encoding="utf-8") as f:
                    abstract_text = f.read()


            # Store data
            data.append({

                "paper_name": file,

                "abstract": abstract_text,

                "content": full_text[:3000]

            })


    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4, ensure_ascii=False)


    print("✅ PDF ingestion completed")
    print("Saved to:", output_file)


# Run independently
if __name__ == "__main__":

    ingest_pdf()