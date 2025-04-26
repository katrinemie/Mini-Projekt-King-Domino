import os
import cv2
import numpy as np
import csv


def classify_tile_by_hsv(tile):
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    avg_hsv = cv2.mean(hsv)[:3]
    hue, saturation, value = avg_hsv

    if 22 < hue < 28 and 229 < saturation < 256 and 155 < value < 206:
        return "Field"
    if 29 < hue < 48 and 82 < saturation < 235 and 38 < value < 81:
        return "Forest"
    if 103 < hue < 110 and 225 < saturation < 256 and 107 < value < 194:
        return "Lake"
    if 34 < hue < 49 and 170 < saturation < 245 and 82 < value < 164:
        return "Grassland"
    if 16 < hue < 27 and 47 < saturation < 179 and 82 < value < 146:
        return "Swamp"
    if 17 < hue < 31 and 29 < saturation < 99 and 19 < value < 56:
        return "Mine"
    if 16 < hue < 44 and 37 < saturation < 163 and 51 < value < 137:
        return "Home"
    
    return "Empty"


def save_tile(tile, terrain_type, board_name, tile_index, output_dir):
    terrain_path = os.path.join(output_dir, terrain_type)
    os.makedirs(terrain_path, exist_ok=True)
    filename = f"{board_name}_tile{tile_index}.jpg"
    cv2.imwrite(os.path.join(terrain_path, filename), tile)


def extract_tiles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f" Kunne ikke indlæse billede: {image_path}")
        return []
    
    tiles = []
    height, width, _ = image.shape
    tile_h = height // 5
    tile_w = width // 5

    for i in range(5):
        for j in range(5):
            tile = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    return tiles


def process_all_boards(input_folder, output_dir):
    print("🔎 Starter behandling af billeder i:", input_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            print(f"  Behandler: {filename}")
            board_name = os.path.splitext(filename)[0]
            tiles = extract_tiles(image_path)
            for idx, tile in enumerate(tiles):
                terrain_type = classify_tile_by_hsv(tile)
                save_tile(tile, terrain_type, board_name, idx, output_dir)

                
                if idx < 2:
                    cv2.imshow(f"{filename} - tile {idx} ({terrain_type})", tile)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
    print(" Færdig med behandling.")


def generate_csv_from_folders(folder_path, output_csv_path):
    rows = []
    for terrain_type in os.listdir(folder_path):
        terrain_dir = os.path.join(folder_path, terrain_type)
        if not os.path.isdir(terrain_dir):
            continue
        for filename in os.listdir(terrain_dir):
            if filename.endswith(".jpg"):
                full_path = os.path.join(terrain_dir, filename)
                rows.append([filename, terrain_type, full_path])

    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "TerrainType", "Path"])
        writer.writerows(rows)


input_folder = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\King Domino dataset\Cropped and perspective corrected boards"
output_folder = r"C:\Users\katri\Desktop\Kingkat"

process_all_boards(input_folder, output_folder)

