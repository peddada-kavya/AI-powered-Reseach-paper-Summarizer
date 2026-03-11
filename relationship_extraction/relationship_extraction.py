import json
import os

# Input entity file
ENTITY_FILE = "../entity_extraction/entities.json"

# Output relationship file
OUTPUT_FILE = "relationships.json"

relationships = []

# Relationship templates
def extract_relationships(entity_record):
    rels = []
    entities = entity_record["entities"]

    persons = [e["entity_text"] for e in entities if e["entity_label"] == "PERSON"]
    orgs = [e["entity_text"] for e in entities if e["entity_label"] == "ORG"]
    dates = [e["entity_text"] for e in entities if e["entity_label"] == "DATE"]
    topics = [e["entity_text"] for e in entities if e["entity_label"] in ["NORP", "EVENT", "WORK_OF_ART"]]

    # Author → Paper
    for person in persons:
        rels.append({
            "subject": person,
            "relation": "WROTE",
            "object": f"Paper_{entity_record['record_id']}"
        })

    # Paper → Organization (Journal / Conference)
    for org in orgs:
        rels.append({
            "subject": f"Paper_{entity_record['record_id']}",
            "relation": "PUBLISHED_IN",
            "object": org
        })

    # Paper → Date
    for date in dates:
        rels.append({
            "subject": f"Paper_{entity_record['record_id']}",
            "relation": "PUBLISHED_ON",
            "object": date
        })

    # Paper → Topic
    for topic in topics:
        rels.append({
            "subject": f"Paper_{entity_record['record_id']}",
            "relation": "HAS_TOPIC",
            "object": topic
        })

    return rels

# Load entity data
if not os.path.exists(ENTITY_FILE):
    print("❌ entities.json not found")
    exit()

with open(ENTITY_FILE, "r", encoding="utf-8") as f:
    entity_data = json.load(f)

# Extract relationships
for record in entity_data:
    record_relationships = extract_relationships(record)
    relationships.extend(record_relationships)

# Save relationships
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(relationships, f, indent=4, ensure_ascii=False)

print("✅ Relationship extraction completed successfully")
print(f"🔗 Total relationships extracted: {len(relationships)}")
