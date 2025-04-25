import cv2 as cv
import numpy as np

# === SETTINGS ===
image_path = "outputs/debug_tiles/tile_final_8.jpg"  # <- RET til en tile med krone
window_name = "HSV Tuner"

# === INITIAL VALUES ===
H_MIN, S_MIN, V_MIN = 20, 100, 100
H_MAX, S_MAX, V_MAX = 35, 255, 255

cv.namedWindow(window_name)
cv.resizeWindow(window_name, 600, 300)

def nothing(x):
    pass

# === TRACKBARS ===
cv.createTrackbar("H Min", window_name, H_MIN, 179, nothing)
cv.createTrackbar("H Max", window_name, H_MAX, 179, nothing)
cv.createTrackbar("S Min", window_name, S_MIN, 255, nothing)
cv.createTrackbar("S Max", window_name, S_MAX, 255, nothing)
cv.createTrackbar("V Min", window_name, V_MIN, 255, nothing)
cv.createTrackbar("V Max", window_name, V_MAX, 255, nothing)

# === LOAD IMAGE ===
image = cv.imread(image_path)
if image is None:
    print("Kunne ikke finde billede:", image_path)
    exit()

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

while True:
    h_min = cv.getTrackbarPos("H Min", window_name)
    h_max = cv.getTrackbarPos("H Max", window_name)
    s_min = cv.getTrackbarPos("S Min", window_name)
    s_max = cv.getTrackbarPos("S Max", window_name)
    v_min = cv.getTrackbarPos("V Min", window_name)
    v_max = cv.getTrackbarPos("V Max", window_name)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(image, image, mask=mask)

    combined = np.hstack((image, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), result))
    cv.imshow(window_name, combined)

    key = cv.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv.destroyAllWindows()
print("Brug disse værdier i count_crowns():")
print("Lower HSV:", lower)
print("Upper HSV:", upper)
