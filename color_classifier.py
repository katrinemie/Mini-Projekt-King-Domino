import os
import glob
import cv2
import numpy as np

# Sti til mappen med billederne 
image_folder = r"/Users/lanjakhorshid/Desktop/Mini-Projekt-King-Domino-5/King Domino dataset/Cropped and perspective corrected boards"

# Definér farveområder (HSV) for King Domino-terræntyper
terrain_color_ranges = {
    "Field": ((20, 100, 100), (30, 255, 255)),  # Gul
    "Lake":        ((90, 100,  50), (130, 255, 255)), # Blå
    "Forest":      ((40,  70,  50), (80,  255, 255)), # Grøn
    "Grassland":   ((35,  40, 120), (70,  200, 255)), # Lysere grøn
    "Swamp":       (( 0,   0,  50), (180,  50, 150)), # Grålige/mørke nuancer
    "Mine":        ((10,  70,  50), (25,  255, 200)), # Brunlige nuancer
    }

#  Find alle billeder i mappen ved hjælp af glob-modulen
image_files = glob.glob(os.path.join(image_folder, "*.[Jj][Pp][Gg]"))
if not image_files:
   print("Ingen billeder fundet i mappen.")
   exit()

# For hvert billede i mappen  
for image_file in image_files:
    img = cv2.imread(image_file)
    if img is None:
        print(f"Kunne ikke læse filen {image_file}")
        continue

    print(f"\nBillede: {os.path.basename(image_file)}")

# Konverter billedet til HSV-farverummet
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Opret en dictionary til at tælle antallet af pixels for hvert terræntype
terrain_counts = {terrain : 0 for terrain in terrain_color_ranges}

# For hvert terrændtype i dictionary, tæl antallet af pixels i billedet
for terrain_name, (lower_vals, upper_vals) in terrain_color_ranges.items():
        # Konverter til NumPy-arrays, som OpenCV kræver
        lower = np.array(lower_vals, dtype=np.uint8)
        upper = np.array(upper_vals, dtype=np.uint8)

        # Lav en maske for pixela, der er inden for farveområdet (Binærts billede med 1 (Hvid) og 0 (sort))
        mask = cv2.inRange(hsv_img, lower, upper)

        # Tæl antallet af pixels, der er 1 (Hvid) i masken
        count = cv2.countNonZero(mask)
        terrain_counts[terrain_name]= count

# Find den terræntype, der har flest pixels  
most_dominant_terrain = max(terrain_counts, key=terrain_counts.get)
max_count = terrain_counts[most_dominant_terrain]

# Print resultaterne
print(f"\nBillede: {os.path.basename(image_file)}")
for terrain_name, count in terrain_counts.items():
    print(f"{terrain_name}: {count}")
print(f"Mest dominerende terræn: {most_dominant_terrain} ({max_count} pixels)")