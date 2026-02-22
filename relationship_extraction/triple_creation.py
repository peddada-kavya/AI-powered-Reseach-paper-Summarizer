import json
import os

# Input & Output files
INPUT_FILE = "../entity_extraction/entities.json"
OUTPUT_FILE = "triples.json"

triples = []

# Load entities.json
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

for record in records:
    paper_id = f"Paper_{record['record_id']}"
    entities = record.get("entities", [])

    for ent in entities:
        text = ent["entity_text"]
        label = ent["entity_label"]

        # Relationship rules
        if label == "PERSON":
            triples.append((paper_id, "HAS_AUTHOR", text))

        elif label == "ORG":
            triples.append((paper_id, "PUBLISHED_IN", text))

        elif label == "DATE":
            triples.append((paper_id, "PUBLISHED_YEAR", text))

        elif label in ["GPE", "LOC"]:
            triples.append((paper_id, "ASSOCIATED_WITH_LOCATION", text))

# Convert tuples to JSON format
triples_json = [
    {
        "subject": t[0],
        "predicate": t[1],
        "object": t[2]
    }
    for t in triples
]

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(triples_json, f, indent=4, ensure_ascii=False)

print("✅ Automatic Triple Creation Completed")
print(f"📦 Total Triples Generated: {len(triples_json)}")
