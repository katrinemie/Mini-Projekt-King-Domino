import os
import cv2
import csv

def load_existing_annotations(csv_path):
    annotations = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                annotations.append(row)
    return annotations

def save_csv(data, csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["image_id", "x", "y", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def annotate_tiles_with_crowns(folder_path, output_csv_path):
    print("Starter manuel annotation.")
    print("0, 1, 2 eller 3 alt efter antal kroner.ESC for at afslutte.")

    if not os.path.exists(folder_path):
        print(f"FEJL: Input-mappen findes ikke: {folder_path}")
        return

    annotated_rows = load_existing_annotations(output_csv_path)

    #koordinater
    x = 0
    y = 0
    tile_size = 100  
    max_row_width = 500  

    for terrain_type in os.listdir(folder_path):
        terrain_dir = os.path.join(folder_path, terrain_type)
        if not os.path.isdir(terrain_dir):
            continue

        for filename in sorted(os.listdir(terrain_dir)):
            if not (filename.endswith(".jpg") or filename.endswith(".png")):
                continue

            full_path = os.path.join(terrain_dir, filename)

            image = cv2.imread(full_path)
            if image is None:
                print(f" Kunne ikke åbne {full_path}")
                continue

            
            cv2.imshow(f"{terrain_type} - {filename}", image)
            while True:
                key = cv2.waitKey(0)

                if key == 27:  
                    print(" Afslutter...")
                    cv2.destroyAllWindows()
                    save_csv(annotated_rows, output_csv_path)
                    return
                elif key in [48, 49, 50, 51]:  # 0, 1, 2, 3
                    crowns = int(chr(key))
                    label = f"{terrain_type} {crowns}"
                    print(f" {filename} → {label}")
                    annotated_rows.append({
                        "image_id": 1,
                        "x": x,
                        "y": y,
                        "label": label
                    })
                    save_csv(annotated_rows, output_csv_path)

                    
                    x += tile_size
                    if x >= max_row_width:
                        x = 0
                        y += tile_size
                    break
                else:
                    print(" Ugyldigt input. kun tasterne 0, 1, 2 eller 3.")

            cv2.destroyAllWindows()

    print("CSV-fil gemt:", output_csv_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "Manuelt opdelt")
    output_csv_path = os.path.join(script_dir, "crowns_annotations.csv")

    annotate_tiles_with_crowns(folder_path, output_csv_path)
