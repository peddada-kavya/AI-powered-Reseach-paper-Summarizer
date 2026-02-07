import pandas as pd
import json

# Load citation Excel file
excel_file = "citation.xlsx"   # change if your file name is different
df = pd.read_excel(excel_file)

# Replace empty cells
df = df.fillna("")

# Convert all values to string (IMPORTANT for JSON)
df = df.astype(str)

# Convert DataFrame to JSON
citation_json = df.to_dict(orient="records")

# Save JSON output
with open("citation.json", "w", encoding="utf-8") as f:
    json.dump(citation_json, f, indent=4, ensure_ascii=False)

print("✅ Citation Excel converted to JSON successfully")
print(f"📦 Total citation records: {len(citation_json)}")
