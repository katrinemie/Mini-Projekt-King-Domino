import cv2 as cv
import numpy as np
import os
import random
import csv
import matplotlib.pyplot as plt

class BoardScorer:
    # Initialiserer BoardScorer objektet med inputfolder, ground truth CSV, output folder, tile størrelse og board dimensioner.
    def __init__(self, input_folder, ground_truth_csv, output_folder="outputs", tile_size=100, rows=5, cols=5):
        self.input_folder = input_folder
        self.ground_truth_csv = ground_truth_csv
        self.output_folder = output_folder
        self.tile_size = tile_size
        self.rows = rows
        self.cols = cols
        os.makedirs(self.output_folder, exist_ok=True)
        self.score_data = []

    # Opdeler et billede i tiles baseret på dets dimensioner og tile størrelse
    def get_tiles(self, image):
        return [[image[y*self.tile_size:(y+1)*self.tile_size, x*self.tile_size:(x+1)*self.tile_size] 
                 for x in range(self.cols)] for y in range(self.rows)]

    # Bestemmer terrain for en tile baseret på dens farve i HSV farverummet
    def get_terrain(self, tile):
        hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        hue, saturation, value = np.median(hsv_tile, axis=(0, 1))
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

    # Tæller antallet af kronede områder (fx blomster eller objekter) i en tile ved at finde konturer i en farvemaskering
    def count_crowns(self, tile):
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

    def find_areas(self, board):
        visited = [[False]*self.cols for _ in range(self.rows)]
        area_map = [[None]*self.cols for _ in range(self.rows)]
        area_scores = {}
        area_id = 0

        # Finder områder (terraintyper) på boardet ved at køre en dybde-først-søgning (DFS) på sammenhængende tiles
        def dfs(r, c, terrain):
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
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

        for r in range(self.rows):
            for c in range(self.cols):
                terrain = board[r][c][0]
                if not visited[r][c] and terrain not in ("Home", "Unknown"):
                    size, crowns, coords = dfs(r, c, terrain)
                    if size > 0:
                        area_scores[area_id] = size * crowns
                        area_id += 1

        return area_map, area_scores

    # Annoterer et billede med områder, scores og andre oplysninger (som kronetælling) og gemmer det som et billede
    def annotate_board(self, image, board, area_map, area_scores, total_score, filename):
        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        overlay = image.copy()
        area_colors = {area_id: (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
                       for area_id in set(filter(lambda x: x is not None, sum(area_map, [])))}

        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c * self.tile_size, r * self.tile_size
                terrain, crowns = board[r][c]
                area_id = area_map[r][c]

                if area_id is not None:
                    color = area_colors[area_id]
                    cv.rectangle(overlay, (x, y), (x + self.tile_size, y + self.tile_size), color, 2)

                if terrain not in ("Home", "Unknown"):
                    text_lines = [f"{terrain}", f"{crowns} crown(s)"]
                    if area_id is not None:
                        text_lines.append(f"A#{area_id} ({area_scores[area_id]}p)")

                    for i, line in enumerate(text_lines):
                        cv.putText(overlay, line, (x + 3, y + 15 + i * 15), font, scale,
                                   (0, 0, 0), thickness + 1, cv.LINE_AA)
                        cv.putText(overlay, line, (x + 3, y + 15 + i * 15), font, scale,
                                   (255, 255, 255), thickness, cv.LINE_AA)

        cv.rectangle(overlay, (0, self.tile_size * self.rows), (self.tile_size * self.cols, self.tile_size * self.rows + 30), (0, 0, 0), -1)
        cv.putText(overlay, f"Total score: {total_score}", (10, self.tile_size * self.rows + 22), font,
                   0.6, (255, 255, 255), 1, cv.LINE_AA)

        return overlay

    # Gemmer score dataene i en CSV-fil
    def save_score_csv(self):
        path = os.path.join(self.output_folder, "scores.csv")
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Score"])
            writer.writerows(self.score_data)

    # Sammenligner de beregnede scores med de sande scores i ground truth CSV'en og beregner fejl
    def compare_with_ground_truth(self):
        ground_truth = {}
        with open(self.ground_truth_csv, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                ground_truth[row["Image"]] = int(row["Score"])

        errors = []
        for filename, pred_score in self.score_data:
            true_score = ground_truth.get(filename, None)
            if true_score is not None:
                error = abs(pred_score - true_score)
                errors.append(error)
                print(f"{filename}: Predicted = {pred_score}, Ground Truth = {true_score}, Error = {error}")

        if errors:
            mae = sum(errors) / len(errors)
            print(f"\nGennemsnitlig fejl (MAE): {mae:.2f}")
        else:
            print("\u26a0\ufe0f Ingen matchende billeder fundet i ground truth CSV.")

    # Kører hele scoring process og gemmer resultaterne i output folderen og CSV fil
    def run(self):
        for filename in sorted(os.listdir(self.input_folder)):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.input_folder, filename)
                image = cv.imread(image_path)
                if image is None:
                    print(f"Kunne ikke indlæse: {filename}")
                    continue

                tiles = self.get_tiles(image)
                board = [[(self.get_terrain(tile), self.count_crowns(tile)) for tile in row] for row in tiles]
                area_map, area_scores = self.find_areas(board)
                total_score = sum(area_scores.values())
                annotated = self.annotate_board(image, board, area_map, area_scores, total_score, filename)

                output_img_path = os.path.join(self.output_folder, f"score_calculator_output_{filename}")
                cv.imwrite(output_img_path, annotated)
                self.score_data.append((filename, total_score))
                print(f"Gemte: {output_img_path} | Score: {total_score}")

        self.save_score_csv()
        print("\nbilleder behandlet og gemt i 'outputs/'")
        print("Scores gemt i: outputs/scores.csv")
        self.compare_with_ground_truth()

if __name__ == "__main__":
    
    scorer = BoardScorer(
        input_folder="splitted_dataset/train/cropped",
        ground_truth_csv="ground_truth_scores.csv",
        output_folder="outputs"
    )
    scorer.run()
