# EdgeDetection.py

import os
import cv2
import numpy as np

# Korrekt sti til din train-cropped mappe
train_cropped_dir = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\King domino mini\Mini-Projekt-King-Domino\splitted_dataset\train\cropped"

# Debugging: Tjek om mappen findes
if not os.path.exists(train_cropped_dir):
    print(f"FEJL: Stien {train_cropped_dir} findes ikke!")
    exit()

# Hent alle filerne i mappen
filenames = os.listdir(train_cropped_dir)

# Loop gennem billeder i train_cropped_dir
for index, filename in enumerate(filenames):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    # Konverter til gråskala
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel Edge Detection
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)

    # Tilføj tekst med filnavnet og billede nummer
    text = f"Fil {index+1}/{len(filenames)}: {filename}"
    cv2.putText(sobel_edges, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Vis billedet med Sobel-kantdetektion
    cv2.imshow("Sobel Edges", sobel_edges)

    # Vent på tastetryk for at gå videre
    key = cv2.waitKey(0)  # 0 betyder vent uendeligt på tastetryk
    if key == 27:  # Hvis ESC-tasten trykkes, lukkes programmet
        print("Programmet blev afsluttet.")
        break

# Luk alle vinduer når programmet stopper
cv2.destroyAllWindows()
