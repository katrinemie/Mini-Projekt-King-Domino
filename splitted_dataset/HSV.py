

import os
import cv2

#path til mappen
train_cropped_dir = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\King domino mini\Mini-Projekt-King-Domino\splitted_dataset\train\cropped"

#om mappen findes
if not os.path.exists(train_cropped_dir):
    print(f"FEJL: Stien {train_cropped_dir} findes ikke!")
    exit()

#Hent bare alle filerne i mappen
filenames = os.listdir(train_cropped_dir)

#Loop gennem billeder
for index, filename in enumerate(filenames):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    #HSV konvertering
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Tilføjer billede tal osv
    text = f"Fil {index+1}/{len(filenames)}: {filename}"
    cv2.putText(image_hsv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #viswr HSV billedet
    cv2.imshow("HSV Billede", image_hsv)

    
    key = cv2.waitKey(0)  
    if key == 27:  
        print("Programmet blev afsluttet.")
        break


cv2.destroyAllWindows()
