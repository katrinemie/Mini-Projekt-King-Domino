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

    # Brug Canny Edge Detection til at finde kanter
    edges = cv2.Canny(gray_image, 100, 200)

    # Find konturer i kantbilledet
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Gennemgå konturerne
    for contour in contours:
        # Hvis konturen er stor nok, forsøger vi at finde en firkant (kroneform)
        if cv2.contourArea(contour) > 500:  # Filtrér små konturer
            # Approximér konturen til en polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Hvis polygonen har 4 hjørner, er det en firkant (som en krone)
            if len(approx) == 4:
                # Tegn en linje rundt om kongekronen
                cv2.drawContours(image, [approx], -1, (0, 255, 255), 3)  # Gul farve (BGR format)

    # Tilføj tekst med filnavnet og billede nummer
    text = f"Fil {index+1}/{len(filenames)}: {filename}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Vis det behandlet billede
    cv2.imshow("Kongekroner Markerede", image)

    # Vent på tastetryk for at gå videre
    key = cv2.waitKey(0)  # 0 betyder vent uendeligt på tastetryk
    if key == 27:  # Hvis ESC-tasten trykkes, lukkes programmet
        print("Programmet blev afsluttet.")
        break

# Luk alle vinduer når programmet stopper
cv2.destroyAllWindows()
