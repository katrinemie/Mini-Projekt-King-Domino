import os
import glob
import cv2 as cv
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Hjælpefunktioner
def get_tiles(image):
    # Eksempel: Del billedet op i 5x5 felter á 100x100 pixels
    # Tilpas tallene 5 og 100 til din situation
    tiles = []
    for y in range(5):
        row_tiles = []
        for x in range(5):
            # Her antager vi, at hver tile er 100 pixels høj og bred
            tile = image[y*100:(y+1)*100, x*100:(x+1)*100]
            row_tiles.append(tile)
        tiles.append(row_tiles)
    return tiles

def get_terrain(tile):
    # Eksempel: brug median HSV for tile
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile.reshape(-1, 3), axis=0)
    
    # Eksempel på simple thresholds
    # Du kan evt. finjustere disse værdier:
    if 21.5 < hue < 27.5 and 225 < saturation < 255.5 and 104 < value < 210:
        return "Field"
    if 25 < hue < 60 and 88.5 < saturation < 247 and 24.5 < value < 78.5:
        return "Forest"
    if 43.5 < hue < 120 and 221.5 < saturation < 275 and 115 < value < 204.5:
        return "Lake"
    if 34.5 < hue < 46.5 and 150 < saturation < 260 and 91.5 < value < 180:
        return "Grassland"
    if 16.5 < hue < 27 and 66 < saturation < 180 and 75 < value < 138.5:
        return "Swamp"
    if 19.5 < hue < 27 and 39.5 < saturation < 150 and 29.4 < value < 80:
        return "Mine"
    # Tilføj flere regler efter behov
    
    return "Unknown"



# 2) Sti til mappen med billederne 
image_folder = "Mini-Projekt-King-Domino-1/splitted_dataset/train/cropped"
# 3) Find alle billeder i mappen
image_files = glob.glob(os.path.join(image_folder, "*.[Jj][Pp][Gg]"))
if not image_files:
    print("Ingen billeder fundet i mappen.")
    exit()

# 4) For hvert billede i mappen  
for image_file in image_files:
    img = cv.imread(image_file)
    if img is None:
        print(f"Kunne ikke læse filen {image_file}")
        continue

    print(f"\nBillede: {os.path.basename(image_file)}")

    # (A) Del billedet op i tiles
    tiles = get_tiles(img)
    
    # (B) Klassificér hver tile
    #     Her kan du enten printe terræntypen for hver tile
    #     eller lave en tælling ligesom i din gamle kode.
    terrain_counts = {}
    
    # Eksempel: gennemløb hver tile i 2D-listen
    for row_idx, row_tiles in enumerate(tiles):
        for col_idx, tile in enumerate(row_tiles):
            predicted_terrain = get_terrain(tile)
            terrain_counts[predicted_terrain] = terrain_counts.get(predicted_terrain, 0) + 1
            
            # Print tile-klassifikationen (valgfrit)
            print(f"Tile[{row_idx}, {col_idx}] => {predicted_terrain}")

    # (C) Find mest dominerende terræn blandt alle tiles i billedet
    #     (Hvis du stadig vil have et "globalt" resultat)
    if len(terrain_counts) > 0 and "Unknown" in terrain_counts:
        # Evt. fjern "Unknown" hvis du ikke vil have den i dominansberegningen
        pass
    
    # Undgå fejlsituation hvor alt er "Unknown"
    if len(terrain_counts) == 0:
        print("Ingen terræntyper genkendt.")
        continue

    most_dominant_terrain = max(terrain_counts, key=terrain_counts.get)
    max_count = terrain_counts[most_dominant_terrain]

    print("\nAntal tiles pr. terræn:")
    for terrain_name, count in terrain_counts.items():
        print(f"{terrain_name}: {count}")
    print(f"Mest dominerende terræn: {most_dominant_terrain} ({max_count} tiles)")