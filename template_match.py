import cv2
import numpy as np

def rotate_image(image, angle):
    """Roter billede omkring centrum uden at klippe det."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # Beregn ny størrelse
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # Juster matrix
    rot_mat[0, 2] += bound_w / 2 - center[0]
    rot_mat[1, 2] += bound_h / 2 - center[1]

    return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), borderValue=(255,255,255))

# === Indlæs billeder ===
board_img = cv2.imread('splitted_dataset/train/cropped/1.jpg')
crown_template = cv2.imread('opdateret_skærmbillede.png')

if board_img is None or crown_template is None:
    raise ValueError("Billede ikke fundet.")

crown_template = cv2.resize(crown_template, (50, 50))

# === Parametre ===
board_height, board_width = board_img.shape[:2]
tile_height = board_height // 5
tile_width = board_width // 5

crown_counts = np.zeros((5, 5), dtype=int)
scales = [0.6, 0.8, 1.0, 1.2]
angles = [0, 90, 180, 270]
threshold = 0.6
highlight_color = (255, 182, 193)

# === Gennemgå tiles ===
for row in range(5):
    for col in range(5):
        x_start = col * tile_width
        y_start = row * tile_height
        tile_bgr = board_img[y_start:y_start + tile_height, x_start:x_start + tile_width]
        total_matches = 0

        for angle in angles:
            rotated_template = rotate_image(crown_template, angle)

            for scale in scales:
                resized_template = cv2.resize(rotated_template, (0, 0), fx=scale, fy=scale)
                h, w = resized_template.shape[:2]

                if h > tile_bgr.shape[0] or w > tile_bgr.shape[1]:
                    continue

                result = cv2.matchTemplate(tile_bgr, resized_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):
                    top_left = (x_start + pt[0], y_start + pt[1])
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(board_img, top_left, bottom_right, highlight_color, 2)
                    total_matches += 1

        crown_counts[row, col] = total_matches
        if total_matches > 0:
            print(f"Tile ({row},{col}) - Fundne kroner: {total_matches}")

# === Vis resultat ===
cv2.imshow("Kroner der bliver markeret", board_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nAntal kroner pr. tile (rækker x kolonner):")
print(crown_counts)
