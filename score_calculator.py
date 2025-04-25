import cv2 as cv
import numpy as np
import os
import random
import csv

# === Indstillinger ===
tile_size = 100
rows, cols = 5, 5

input_folder = "splitted_dataset/train/cropped"
output_folder = "outputs"
debug_folder = os.path.join(output_folder, "debug_tiles")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# === Indlæs krone-template ===
crown_template = cv.imread("opdateret_skærmbillede.png")
if crown_template is None:
    raise FileNotFoundError("Kunne ikke finde krone-template: opdateret_skærmbillede.png")

# === Funktioner ===
def get_tiles(image):
    return [[image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] for x in range(cols)] for y in range(rows)]

    def get_terrain(self, tile):
        hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        hue, saturation, value = np.median(hsv_tile, axis=(0,1))

        if 24 < hue < 27.5 and 225 < saturation < 255 and 104 < value < 210:
            return "Field"
        if 25 < hue < 60 and 97 < saturation < 247 and 30 < value < 70:
            return "Forest"
        if 43.5 < hue < 120 and 223.5 < saturation < 275 and 115 < value < 190:
            return "Lake"
        if 36 < hue < 45.5 and 150 < saturation < 260 and 100 < value < 180:
            return "Grassland"
        if 21 < hue < 27 and 66 < saturation < 180 and 75 < value < 135:
            return "Swamp"
        if 20 < hue < 27 and 60 < saturation < 150 and 30 < value < 80:
            return "Mine"
        if 18 < hue < 35 and 166 < saturation < 225 and 100 < value < 160:
            return "Table"
        return "Home"

def non_max_suppression(centers, min_dist=15):
    final_centers = []
    for cx, cy in centers:
        if all(np.hypot(cx - fx, cy - fy) > min_dist for fx, fy in final_centers):
            final_centers.append((cx, cy))
    return final_centers

def count_crowns(tile):
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    lower_yellow = np.array([25, 130, 130])
    upper_yellow = np.array([32, 255, 255])
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centers = [tuple(np.mean(cnt.reshape(-1, 2), axis=0).astype(int))
               for cnt in contours if 100 < cv.contourArea(cnt) < 800]
    final_centers = non_max_suppression(centers)
    return len(final_centers)

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

    def annotate_board(self, image, board, area_map, area_scores, total_score):
        font = cv.FONT_HERSHEY_SIMPLEX
        overlay = image.copy()
        area_colors = {area_id: (random.randint(60,255), random.randint(60,255), random.randint(60,255)) for area_id in set(filter(None.__ne__, sum(area_map, [])))}

        print("\n=== Områder fundet ===")
        for area_id, score in area_scores.items():
            size = sum(row.count(area_id) for row in area_map)
            crowns = score // size if size > 0 else 0
            print(f"A#{area_id}: {size} felter, {crowns} kroner → {score}p")

        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c * self.tile_size, r * self.tile_size
                terrain, crowns = board[r][c]
                area_id = area_map[r][c]
                if area_id is not None:
                    cv.rectangle(overlay, (x, y), (x + self.tile_size, y + self.tile_size), area_colors[area_id], 2)

                if terrain not in ("Home", "Table"):
                    texts = [f"{terrain}", f"{crowns} crown(s)"]
                    if area_id is not None:
                        texts.append(f"A#{area_id} ({area_scores[area_id]}p)")
                    for i, line in enumerate(texts):
                        cv.putText(overlay, line, (x + 3, y + 15 + i * 15), font, 0.4, (0, 0, 0), 2, cv.LINE_AA)
                        cv.putText(overlay, line, (x + 3, y + 15 + i * 15), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        cv.rectangle(overlay, (0, self.tile_size * self.rows), (self.tile_size * self.cols, self.tile_size * self.rows + 30), (0, 0, 0), -1)
        cv.putText(overlay, f"Total score: {total_score}", (10, self.tile_size * self.rows + 22), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        return overlay

def save_score_csv(score_data, csv_path="outputs/scores.csv"):
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Score"])
        writer.writerows(score_data)

# === HOVEDKØRSEL ===
score_data = []

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        image = cv.imread(image_path)
        tiles = self.get_tiles(image)
        board = [[(self.get_terrain(tile), self.count_crowns(tile)) for tile in row] for row in tiles]

        area_map, area_scores = self.find_areas(board)
        total_score = sum(area_scores.values())
        annotated = annotate_board(image, board, area_map, area_scores, total_score, filename)

        output_img_path = os.path.join(output_folder, f"score_calculator_output_{filename}")
        cv.imwrite(output_img_path, annotated)
        score_data.append((filename, total_score))
        print(f"✔ Gemte: {output_img_path} | Score: {total_score}")

save_score_csv(score_data)
print("\n✔ Alle billeder behandlet og gemt i 'outputs/'")
print("✔ Scores gemt i: outputs/scores.csv")