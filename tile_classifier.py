import cv2
import numpy as np
import os
import glob

class TileClassifier:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        if not self.image_paths:
            raise FileNotFoundError(" Ingen billeder fundet i mappen.")

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
        return "Unknown"

    def split_to_tiles(self, image):
        h, w = image.shape[:2]
        tile_h = h // 5
        tile_w = w // 5
        return [[image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w] for x in range(5)] for y in range(5)]

    def annotate_tiles(self, image, labels):
        annotated = image.copy()
        h, w = image.shape[:2]
        tile_h = h // 5
        tile_w = w // 5

        for row in range(5):
            for col in range(5):
                x = col * tile_w
                y = row * tile_h
                label = labels[row][col]
                cv2.putText(annotated, label, (x + 5, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.rectangle(annotated, (x, y), (x + tile_w, y + tile_h), (0, 255, 0), 1)
        return annotated

    def process_images(self):
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f" Kunne ikke læse billedet: {path}")
                continue

            tiles = self.split_to_tiles(img)
            terrain_labels = []

            print(f"\n📄 Billede: {os.path.basename(path)}")
            for row_idx, row in enumerate(tiles):
                label_row = []
                for col_idx, tile in enumerate(row):
                    label = self.classify_tile(tile)
                    label_row.append(label)
                    print(f"Tile[{row_idx},{col_idx}] = {label}")
                terrain_labels.append(label_row)

            annotated = self.annotate_tiles(img, terrain_labels)
            cv2.imshow("Klassificerede Tiles", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier = TileClassifier(input_folder='splitted_dataset/train/cropped')
    classifier.process_images()
