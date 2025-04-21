import cv2
import numpy as np
import os
import glob

# === Klassifikation med HSV-farverum ===
def classify_tile(tile):
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    hue, sat, val = np.median(hsv.reshape(-1, 3), axis=0)

    # Thresholds baseret på typiske værdier – du kan justere!
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

# === Funktion: del billede i 5x5 tiles
def split_to_tiles(image):
    h, w = image.shape[:2]
    tile_h = h // 5
    tile_w = w // 5
    tiles = []
    for y in range(5):
        row = []
        for x in range(5):
            tile = image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w]
            row.append(tile)
        tiles.append(row)
    return tiles

# === Visualisering med klassificering ===
def annotate_tiles(image, labels):
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

# === MAIN ===
if __name__ == "__main__":
    input_folder = 'splitted_dataset/train/cropped'
    image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))

    if not image_paths:
        print("⚠️ Ingen billeder fundet.")
        exit()

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print("⚠️ Kunne ikke læse billedet:", path)
            continue

        tiles = split_to_tiles(img)
        terrain_labels = []

        print(f"\n🧩 Billede: {os.path.basename(path)}")
        for row_idx, row in enumerate(tiles):
            label_row = []
            for col_idx, tile in enumerate(row):
                label = classify_tile(tile)
                label_row.append(label)
                print(f"Tile[{row_idx},{col_idx}] = {label}")
            terrain_labels.append(label_row)

        annotated = annotate_tiles(img, terrain_labels)
        cv2.imshow("Klassificerede Tiles", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
