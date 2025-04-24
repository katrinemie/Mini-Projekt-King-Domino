import os
import csv
import cv2

def visualize_ground_truth(csv_path, tile_folder, output_folder=None, show_images=True):
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            x = int(row["x"])
            y = int(row["y"])
            terrain = row["terrain"]
            crowns = int(row["crowns"])

            filename = f"tile_{image_id}_{x}_{y}.jpg"
            filepath = os.path.join(tile_folder, filename)

            if not os.path.exists(filepath):
                print(f" Mangler billede: {filename}")
                continue

            image = cv2.imread(filepath)
            if image is None:
                print(f" Kunne ikke åbne: {filepath}")
                continue

            # Forbedret label-stil nederst
            label = f"{terrain} ({crowns})"
            label_position = (5, image.shape[0] - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Semi-transparent baggrund
            overlay = image.copy()
            cv2.rectangle(overlay, (0, image.shape[0] - 20), (image.shape[1], image.shape[0]), (255, 255, 255), -1)
            alpha = 0.5
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Tekst
            cv2.putText(image, label, label_position, font, 0.5, (0, 0, 0), 1)

            # Vis eller gem
            if show_images:
                cv2.imshow(f"{filename}", image)
                key = cv2.waitKey(500)
                if key == 27:  # ESC
                    break
                cv2.destroyAllWindows()
            elif output_folder:
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)

    print("Færdig med visualisering.")

# === Brug det her ===
csv_path = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\ground_truth_split.csv"
tile_folder = r"C:\Users\katri\Desktop\Kingkat\All_Tiles"
output_folder = r"C:\Users\katri\Desktop\Kingkat\Visualiseret"

visualize_ground_truth(csv_path, tile_folder, output_folder=output_folder, show_images=False)

