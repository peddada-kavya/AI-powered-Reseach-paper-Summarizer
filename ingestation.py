import os
import json
from PyPDF2 import PdfReader

pdf_folder = "pdf_files"
abstract_folder = "abstracts"
output_file = "ingested_data.json"

data = []

for file in os.listdir(pdf_folder):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)

        # Extract PDF text
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            if page.extract_text():
                full_text += page.extract_text()

        # Read abstract
        abstract_file = file.replace(".pdf", ".txt")
        abstract_path = os.path.join(abstract_folder, abstract_file)

        abstract_text = ""
        if os.path.exists(abstract_path):
            with open(abstract_path, "r", encoding="utf-8") as f:
                abstract_text = f.read()

        data.append({
            "paper_name": file,
            "abstract": abstract_text,
            "content": full_text[:2000]
        })

# Save JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ PDF + Abstract ingestion completed")
