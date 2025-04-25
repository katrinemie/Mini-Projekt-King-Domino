import csv

def load_classifications(csv_path, is_ground_truth=False):
    data = {}
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_ground_truth:
                key = (row["image_id"], int(row["x"]), int(row["y"]))
                data[key] = row["terrain"]
            else:
                filename = row["filename"].replace(".jpg", "")
                image_id = filename
                key = (image_id, int(row["column"]) * 100, int(row["row"]) * 100)
                data[key] = row["classification"]
    return data

def compare_classifications(gt_path, pred_path):
    ground_truth = load_classifications(gt_path, is_ground_truth=True)
    predictions = load_classifications(pred_path)

    correct = 0
    total = 0

    for key, true_label in ground_truth.items():
        pred_label = predictions.get(key)
        if pred_label:
            total += 1
            if pred_label == true_label:
                correct += 1
        else:
            print(f" Mangler forudsigelse for tile: {key}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

# === Paths til CSV-filer ===
ground_truth_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\ground_truth_train_split.csv"
predictions_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\tile_classifications.csv"

compare_classifications(ground_truth_csv, predictions_csv)
