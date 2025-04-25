import cv2 as cv
import numpy as np
import os
import random
import csv

# === Indstillinger ===
tile_size = 100
rows, cols = 5, 5

input_folder = "splitted_dataset/train/cropped"
ground_truth_csv = "ground_truth_scores.csv"
output_folder = "outputs"
debug_folder = os.path.join(output_folder, "debug_tiles")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# === Funktioner ===
def get_tiles(image):
    return [[image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] for x in range(cols)] for y in range(rows)]

def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile.reshape(-1, 3), axis=0)

    if 21.5 < hue < 27.5 and 225 < saturation < 255 and 104 < value < 210:
        return "Field"
    if 25 < hue < 60 and 88 < saturation < 247 and 24 < value < 78:
        return "Forest"
    if 90 < hue < 130 and 100 < saturation < 255 and 100 < value < 230:
        return "Lake"
    if 34 < hue < 46 and 150 < saturation < 255 and 90 < value < 180:
        return "Grassland"
    if 16 < hue < 27 and 66 < saturation < 180 and 75 < value < 140:
        return "Swamp"
    if 19 < hue < 27 and 39 < saturation < 150 and 29 < value < 80:
        return "Mine"
    if saturation < 60 and 60 < value < 200:
        return "Home"
    return "Unknown"

def count_crowns(tile):
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 140, 140])
    upper_yellow = np.array([33, 255, 255])
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    crown_count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 80 < area < 800:
            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.3:
                crown_count += 1
    return crown_count

def find_areas(board):
    visited = [[False]*cols for _ in range(rows)]
    area_map = [[None]*cols for _ in range(rows)]
    area_scores = {}
    area_id = 0

    def dfs(r, c, terrain):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0, 0, []
        if visited[r][c] or board[r][c][0] != terrain:
            return 0, 0, []
        visited[r][c] = True
        area_map[r][c] = area_id
        count = 1
        crowns = board[r][c][1]
        coords = [(r, c)]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            a, b, c_list = dfs(r+dr, c+dc, terrain)
            count += a
            crowns += b
            coords += c_list
        return count, crowns, coords

    for r in range(rows):
        for c in range(cols):
            terrain = board[r][c][0]
            if not visited[r][c] and terrain not in ("Home", "Unknown"):
                size, crowns, coords = dfs(r, c, terrain)
                if size > 0:
                    area_scores[area_id] = size * crowns
                    area_id += 1

    return area_map, area_scores

def annotate_board(image, board, area_map, area_scores, total_score, filename):
    font = cv.FONT_HERSHEY_SIMPLEX
    overlay = image.copy()
    for r in range(rows):
        for c in range(cols):
            x, y = c * tile_size, r * tile_size
            terrain, crowns = board[r][c]
            cv.putText(overlay, f"{terrain[:2]}-{crowns}", (x + 10, y + 50), font, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(overlay, f"Total score: {total_score}", (10, tile_size * rows + 22), font,
               0.6, (255, 255, 255), 1, cv.LINE_AA)
    return overlay

def save_score_csv(score_data, csv_path="outputs/scores.csv"):
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Score"])
        writer.writerows(score_data)

def compare_with_ground_truth(predicted_scores, ground_truth_csv):
    ground_truth = {}
    with open(ground_truth_csv, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row.get("Image") or row.get("image") or row.get("image_id")
            score_val = row.get("Score") or row.get("score")
            if image_name and score_val:
                ground_truth[image_name] = int(score_val)

    errors = []
    for filename, pred_score in predicted_scores:
        true_score = ground_truth.get(filename)
        if true_score is not None:
            error = abs(pred_score - true_score)
            errors.append(error)
            print(f"{filename}: Predicted = {pred_score}, Ground Truth = {true_score}, Error = {error}")

    if errors:
        mae = sum(errors) / len(errors)
        print(f"\n📊 Gennemsnitlig fejl (MAE): {mae:.2f}")
    else:
        print("⚠️ Ingen matchende billeder fundet i ground truth CSV.")

# === HOVEDKØRSEL ===
score_data = []

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        image = cv.imread(image_path)
        if image is None:
            print(f"❌ Kunne ikke indlæse: {filename}")
            continue

        tiles = get_tiles(image)
        board = [[(get_terrain(tile), count_crowns(tile)) for tile in row] for row in tiles]
        area_map, area_scores = find_areas(board)
        total_score = sum(area_scores.values())
        annotated = annotate_board(image, board, area_map, area_scores, total_score, filename)

        output_img_path = os.path.join(output_folder, f"score_calculator_output_{filename}")
        cv.imwrite(output_img_path, annotated)
        score_data.append((filename, total_score))
        print(f"✔ Gemte: {output_img_path} | Score: {total_score}")

save_score_csv(score_data)
print("\n✔ Alle billeder behandlet og gemt i 'outputs/'")
print("✔ Scores gemt i: outputs/scores.csv")

compare_with_ground_truth(score_data, ground_truth_csv)
