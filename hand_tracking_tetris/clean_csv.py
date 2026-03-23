import csv
import re

input_file = "gestures_control_processed1.csv"  # your current file
output_file = "gestures_control_processed_clean.csv"  # where you want the cleaned data
# Read the entire file as one string (to handle multi-line entries)
with open(input_file, "r", encoding="utf-8") as infile:
    data = infile.read()

# Split the file into entries (each ending with ',label,label')
entries = re.findall(r'"\[(.*?)\]",\s*([A-Za-z0-9_]+),[A-Za-z0-9_]+', data, re.DOTALL)

cleaned_rows = []

for nums_str, label in entries:
    # Remove newlines and split by spaces
    nums = [n for n in nums_str.replace("\n", " ").split() if n]
    cleaned_rows.append(nums + [label])

# Write everything to a proper CSV
with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cleaned_rows)

print(f"✅ Done! {len(cleaned_rows)} samples saved to '{output_file}'.")