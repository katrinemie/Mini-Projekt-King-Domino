import cv2
import numpy as np
import os
import glob

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    rot_mat[0, 2] += bound_w / 2 - center[0]
    rot_mat[1, 2] += bound_h / 2 - center[1]
    return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), borderValue=(255, 255, 255))

# === Parametre ===
input_folder = 'splitted_dataset/train/cropped'
template_path = 'opdateret_skærmbillede.png'
output_folder = 'outputs_with_crowns'
threshold = 0.57
scales = [0.6, 0.8, 1.0, 1.2]
angles = [0, 90, 180, 270]

# === Opret output mappe hvis den ikke findes ===
os.makedirs(output_folder, exist_ok=True)

# === Indlæs template ===
template = cv2.imread(template_path)
if template is None:
    raise ValueError("Template billede ikke fundet.")
template = cv2.resize(template, (50, 50))

# === Loop gennem alle billeder ===
image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
for path in image_paths:
    filename = os.path.basename(path)
    print(f"Behandler: {filename}")
    board_img = cv2.imread(path)
    if board_img is None:
        print(f"Kunne ikke indlæse {filename}")
        continue

    board_copy = board_img.copy()
    board_height, board_width = board_img.shape[:2]
    tile_height = board_height // 5
    tile_width = board_width // 5
    highlight_color = (255, 182, 193)

    for row in range(5):
        for col in range(5):
            x_start = col * tile_width
            y_start = row * tile_height
            tile = board_img[y_start:y_start + tile_height, x_start:x_start + tile_width]

            for angle in angles:
                rotated_template = rotate_image(template, angle)

                for scale in scales:
                    scaled_template = cv2.resize(rotated_template, (0, 0), fx=scale, fy=scale)
                    h, w = scaled_template.shape[:2]

                    if h > tile.shape[0] or w > tile.shape[1] or h < 15 or w < 15:
                        continue

                    result = cv2.matchTemplate(tile, scaled_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= threshold)

                    match_rects = [[pt[0], pt[1], w, h] for pt in zip(*locations[::-1])]
                    rects, _ = cv2.groupRectangles(match_rects, 1, 0.5)
                    rects = rects[:3]

                    for (x, y, w, h) in rects:
                        top_left = (x_start + x, y_start + y)
                        bottom_right = (top_left[0] + w, top_left[1] + h)
                        cv2.rectangle(board_copy, top_left, bottom_right, highlight_color, 2)

    # === Vis billede ===
    cv2.imshow(f"Kroner – {filename}", board_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # === Gem resultat ===
    output_path = os.path.join(output_folder, f"marked_{filename}")
    cv2.imwrite(output_path, board_copy)
    print(f"Gemte: {output_path}")
