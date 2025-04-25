import csv

def split_labels_and_generate_ground_truth(input_csv, output_csv):
    rows = []

    with open(input_csv, newline='', encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            label = row["label"].strip()

            # Split label: fx "Forest 2"
            if " " in label:
                terrain, crowns = label.rsplit(" ", 1)
            else:
                terrain, crowns = label, "0"

            rows.append([
                row["image_id"],
                row["x"],
                row["y"],
                terrain,
                int(crowns)
            ])

    with open(output_csv, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["image_id", "x", "y", "terrain", "crowns"])
        writer.writerows(rows)

    print(f"✅ Ny ground truth CSV gemt: {output_csv}")

# === Korrekte stier ===
input_csv = r"tiles_crowns.csv"
output_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\ground_truth_split.csv"

split_labels_and_generate_ground_truth(input_csv, output_csv)
