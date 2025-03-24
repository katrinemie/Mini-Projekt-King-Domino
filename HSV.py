import cv2
import os

train_dir = "train"  

for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    image = cv2.imread(img_path)  
    
    if image is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    cv2.imshow("Original Billede", image)
    cv2.waitKey(500)  
cv2.destroyAllWindows()

#Starter HSV

for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Laver det til HSV

    cv2.imshow("HSV Billede", image_hsv)
    cv2.waitKey(500)
cv2.destroyAllWindows()


#Thresholding

for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Gråskala
    _, thresholded = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)  #thhhreshold

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  

    cv2.imshow("Fundet countous", image)
    cv2.waitKey(500)
cv2.destroyAllWindows()


#Mean kernel


import numpy as np

for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    kernel = np.ones((5,5), np.float32) / 25  
    image_blur = cv2.filter2D(image, -1, kernel)

    cv2.imshow("blurrd billede", image_blur)
    cv2.waitKey(500)
cv2.destroyAllWindows()


#Kanter sobel


for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    cv2.imshow("Sobel edges", edges)
    cv2.waitKey(500)
cv2.destroyAllWindows()