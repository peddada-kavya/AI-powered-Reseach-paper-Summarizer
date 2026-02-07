import pandas as pd
import json

# Load references Excel file
excel_file = "reference.xlsx"   # change if file name differs
df = pd.read_excel(excel_file)

# Replace NaN with empty string
df = df.fillna("")

# Convert all values to string (avoids Timestamp / type errors)
df = df.astype(str)

# Convert to JSON format
references_json = df.to_dict(orient="records")

# Save JSON output
with open("reference.json", "w", encoding="utf-8") as f:
    json.dump(references_json, f, indent=4, ensure_ascii=False)

print("✅ References Excel converted to JSON successfully")
print(f"📦 Total reference: {len(references_json)}")
