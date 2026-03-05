import json
import spacy
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input JSON files (relative paths)
INPUT_FILES = {
    "metadata": "../papers/Metadata.json",
    "citations": "../citation/citation.json",
    "references": "../references/reference.json",
    "ingested": "../ingested_data.json"
}

OUTPUT_FILE = "entities.json"

all_entities = []

def extract_entities(text, source, record_id):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "entity_text": ent.text,
            "entity_label": ent.label_
        })

    return {
        "source": source,
        "record_id": record_id,
        "entities": entities
    }

# Process each input file
for source_name, file_path in INPUT_FILES.items():
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, record in enumerate(data):
        combined_text = " ".join(str(v) for v in record.values())
        entity_data = extract_entities(combined_text, source_name, idx)
        all_entities.append(entity_data)

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_entities, f, indent=4, ensure_ascii=False)

print("✅ Entity extraction completed successfully")
print(f"📦 Total records processed: {len(all_entities)}")
