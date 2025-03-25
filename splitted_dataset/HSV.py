import os
import cv2
import numpy as np

# Korrekt sti til din train-cropped mappe
train_cropped_dir = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\King domino mini\Mini-Projekt-King-Domino\splitted_dataset\train\cropped"

# Debugging: Tjek om mappen findes
if not os.path.exists(train_cropped_dir):
    print(f"FEJL: Stien {train_cropped_dir} findes ikke!")
    exit()

# Loop gennem billeder i train_cropped_dir
for filename in os.listdir(train_cropped_dir):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    cv2.imshow("Original Billede", image)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# HSV-konvertering
for filename in os.listdir(train_cropped_dir):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV Billede", image_hsv)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# Thresholding
for filename in os.listdir(train_cropped_dir):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Fundne Contours", image)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# Mean Kernel Blur
kernel = np.ones((5,5), np.float32) / 25  

for filename in os.listdir(train_cropped_dir):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_blur = cv2.filter2D(image, -1, kernel)

    cv2.imshow("Blurred Billede", image_blur)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# Sobel Edge Detection
for filename in os.listdir(train_cropped_dir):
    img_path = os.path.join(train_cropped_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    cv2.imshow("Sobel Edges", edges)
    cv2.waitKey(500)

cv2.destroyAllWindows()
