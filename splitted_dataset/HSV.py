import os
import cv2
import numpy as np

# Paths
train_cropped_dir = r"//Applications/Desktop/Mini-Projekt-King-Domino-1/Mini-Projekt-King-Domino-1/splitted_dataset/train/cropped"
reference_crown_path = r"/Applications/Desktop/Mini-Projekt-King-Domino-1/splitted_dataset/train/reference/krone.png"

# Check hvis mappen findes
if not os.path.exists(train_cropped_dir):
    print(f"FEJL: Stien {train_cropped_dir} findes ikke!")
    exit()

# Load reference krone
template = cv2.imread(reference_crown_path)
if template is None:
    print("Kunne ikke indlæse reference krone.")
    exit()

# Konverter til HSV
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

# Størrelse på template
w, h = template.shape[1], template.shape[0]

# Loop gennem cropped billeder
filenames = os.listdir(train_cropped_dir)

for index, filename in enumerate(filenames):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    # Konverter billede til HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Template Matching i HSV (matcher kun V-kanalen)
    result = cv2.matchTemplate(image_hsv[:, :, 2], template_hsv[:, :, 2], cv2.TM_CCOEFF_NORMED)

    # Threshold
    threshold = 0.75
    locations = np.where(result >= threshold)

    # Tegn bokse
    for pt in zip(*locations[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    # Tilføj tekst
    text = f"Fil {index+1}/{len(filenames)}: {filename}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Vis resultat
    cv2.imshow("Resultat", image)

    key = cv2.waitKey(0)
    if key == 27:  # Esc
        print("Programmet blev afsluttet.")
        break

cv2.destroyAllWindows()





