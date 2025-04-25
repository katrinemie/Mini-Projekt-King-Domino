import csv

# === Paths ===
ground_truth_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\ground_truth.csv"
tile_predictions_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\tile_classifications.csv"
output_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\ground_truth_train_split.csv"

# === Hent image_ids fra predictions ===
used_image_ids = set()
with open(tile_predictions_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row["filename"]
        if filename.endswith(".jpg"):
            image_id = filename.replace(".jpg", "")
            used_image_ids.add(image_id)

# === Filtrer ground truth ===
filtered_rows = []
with open(ground_truth_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["image_id"] in used_image_ids:
            filtered_rows.append(row)

# === Gem ny ground truth ===
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    fieldnames = ["image_id", "x", "y", "terrain", "crowns"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

print(f"Filtreret ground truth gemt som: {output_csv}")
print(f"Antal gemte rækker: {len(filtered_rows)}")
