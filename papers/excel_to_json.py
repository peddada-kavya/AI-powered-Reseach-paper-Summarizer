import pandas as pd
import json

# Load Excel file
excel_file = "metadata.xlsx"   # change name if needed
df = pd.read_excel(excel_file)

# Replace NaN with empty string
df = df.fillna("")

# Convert all values to string (IMPORTANT FIX)
df = df.astype(str)

# Convert to JSON
json_data = df.to_dict(orient="records")

# Save JSON
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print("✅ Metadata Excel converted to JSON successfully")
print(f"📦 Total records: {len(json_data)}")
