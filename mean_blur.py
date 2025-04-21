# MeanBlur.py

import os
import cv2
import numpy as np

# Korrekt sti til din train-cropped mappe
train_cropped_dir = "King Domino dataset/Cropped and perspective corrected boards"
# Debugging: Tjek om mappen findes
if not os.path.exists(train_cropped_dir):
    print(f"FEJL: Stien {train_cropped_dir} findes ikke!")
    exit()

# Hent alle filerne i mappen
filenames = os.listdir(train_cropped_dir)

# Mean Kernel Blur
kernel = np.ones((5,5), np.float32) / 25  

# Loop gennem billeder i train_cropped_dir
for index, filename in enumerate(filenames):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    # Apply Mean Kernel Blur
    image_blur = cv2.filter2D(image, -1, kernel)

    # Tilføj tekst med filnavnet og billede nummer
    text = f"Fil {index+1}/{len(filenames)}: {filename}"
    cv2.putText(image_blur, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Vis det uskarpe billede
    cv2.imshow("Blurred Billede", image_blur)

    # Vent på tastetryk for at gå videre
    key = cv2.waitKey(0)  # 0 betyder vent uendeligt på tastetryk
    if key == 27:  # Hvis ESC-tasten trykkes, lukkes programmet
        print("Programmet blev afsluttet.")
        break

# Luk alle vinduer når programmet stopper
cv2.destroyAllWindows()
