import cv2
import numpy as np
import os
import glob
import pandas as pd

class TileAnalyzer:
    def __init__(self, input_folder, ground_truth_csv):
        self.input_folder = input_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        if not self.image_paths:
            raise FileNotFoundError(" Ingen billeder fundet i mappen.")
        self.ground_truth = self.load_ground_truth(ground_truth_csv)

    def load_ground_truth(self, csv_path):
        df = pd.read_csv(csv_path)
        ground_truth = {}
        for img_id in df['image_id'].unique():
            matrix = [['' for _ in range(5)] for _ in range(5)]
            subset = df[df['image_id'] == img_id]
            for _, row in subset.iterrows():
                col = int(row['x'] // 100)
                r = int(row['y'] // 100)
                matrix[r][col] = row['terrain']
            ground_truth[f"{img_id}.jpg"] = matrix
        return ground_truth

    def classify_tile(self, tile):
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        hue, sat, val = np.median(hsv.reshape(-1, 3), axis=0)

        if 21.5 < hue < 27.5 and 225 < sat < 255 and 104 < val < 210:
            return "Field"
        if 25 < hue < 60 and 88 < sat < 247 and 24 < val < 78:
            return "Forest"
        if 90 < hue < 130 and 100 < sat < 255 and 100 < val < 230:
            return "Lake"
        if 34 < hue < 46 and 150 < sat < 255 and 90 < val < 180:
            return "Grassland"
        if 16 < hue < 27 and 66 < sat < 180 and 75 < val < 140:
            return "Swamp"
        if 19 < hue < 27 and 39 < sat < 150 and 29 < val < 80:
            return "Mine"
        if sat < 60 and 60 < val < 200:
            return "Home"
        if 18 < hue < 35 and 166 < sat < 225 and 100 < val < 160:
            return "Table"
        return "Unknown"

    def split_to_tiles(self, image):
        h, w = image.shape[:2]
        tile_h = h // 5
        tile_w = w // 5
        return [[image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w] for x in range(5)] for y in range(5)]

    def process_images(self):
        total_tiles = 0
        correct_tiles = 0

        for path in self.image_paths:
            filename = os.path.basename(path)
            if filename not in self.ground_truth:
                print(f" Ingen ground truth for {filename}")
                continue

            img = cv2.imread(path)
            if img is None:
                print(f" Kunne ikke læse billedet: {filename}")
                continue

            tiles = self.split_to_tiles(img)
            predicted_labels = [[self.classify_tile(tile) for tile in row] for row in tiles]
            true_labels = self.ground_truth[filename]

            for r in range(5):
                for c in range(5):
                    total_tiles += 1
                    if predicted_labels[r][c] == true_labels[r][c]:
                        correct_tiles += 1
                    else:
                        print(f" {filename} - Tile ({r},{c}): Forventet {true_labels[r][c]}, Fundet {predicted_labels[r][c]}")

        accuracy = (correct_tiles / total_tiles) * 100 if total_tiles else 0
        print(f"\n Samlet Neighbour Detection Nøjagtighed: {accuracy:.2f}%")

# === Main ===
if __name__ == "__main__":
    analyzer = TileAnalyzer(
        input_folder='splitted_dataset/train/cropped',
        ground_truth_csv='ground_truth_split.csv'
    )
    analyzer.process_images()
