import csv

file1 = "gestures_control.csv"           # original
file2 = "gestures_control_processed_clean.csv" # augmented
merged_file = "gestures_control_merged_all.csv"

with open(file1, newline="") as f1, open(file2, newline="") as f2, open(merged_file, "w", newline="") as out:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(out)

    # write all rows from first CSV
    for row in reader1:
        writer.writerow(row)

    # write all rows from second CSV
    for row in reader2:
        writer.writerow(row)

print(f"Merged files into {merged_file}")
