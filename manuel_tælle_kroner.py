import os
import cv2
import csv


def load_existing_annotations(csv_path):
    existing = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing[row["Filename"]] = row
    return existing

 
def save_csv(rows, output_csv_path):
    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "TerrainType", "Crowns"])
        writer.writerows(rows)

#Hovedfunktion til annotation
def annotate_tiles_with_crowns(folder_path, output_csv_path):
    print(" Starter manuel annotation.")
    print("Tryk 0, 1, 2 eller 3 alt efter antal kroner. Tryk ESC for at afslutte.")

    #Hent eksisterende annotationer
    existing_annotations = load_existing_annotations(output_csv_path)
    annotated_rows = list(existing_annotations.values())

    for terrain_type in os.listdir(folder_path):
        terrain_dir = os.path.join(folder_path, terrain_type)
        if not os.path.isdir(terrain_dir):
            continue

        for filename in sorted(os.listdir(terrain_dir)):
            if not filename.endswith(".jpg") and not filename.endswith(".png"):
                continue
            if filename in existing_annotations:
                print(f" Springer over allerede annoteret: {filename}")
                continue

            full_path = os.path.join(terrain_dir, filename)
            image = cv2.imread(full_path)
            if image is None:
                print(f"øvvv Kunne ikke åbne {full_path}")
                continue

            #vis billedet
            cv2.imshow(f"{terrain_type} - {filename}", image)
            while True:
                key = cv2.waitKey(0)

                if key in [27]:  # ESC
                    print(" Afslutter")
                    cv2.destroyAllWindows()
                    save_csv(annotated_rows, output_csv_path)
                    return
                elif key in [48, 49, 50, 51]:  # 0, 1, 2, 3
                    crowns = int(chr(key))
                    print(f" {filename} → {crowns} kroner")
                    annotated_rows.append([filename, terrain_type, crowns])
                    break
                else:
                    print("Ugyldigt input. Brug kun tasterne 0, 1, 2 eller 3.")

            cv2.destroyAllWindows()

    save_csv(annotated_rows, output_csv_path)
    print(" CSV-fil gemt:", output_csv_path)


input_folder = r"C:\Users\katri\Desktop\Kingkat"
output_csv = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Opdelt_terræn\ground_truth.csv"

annotate_tiles_with_crowns(input_folder, output_csv)
