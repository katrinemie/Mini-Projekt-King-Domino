import cv2 as cv
import numpy as np
import os
import random
import pandas as pd

class ScoreCalculator:
    def __init__(self, tile_size=100, rows=5, cols=5):
        self.tile_size = tile_size
        self.rows = rows
        self.cols = cols

    def get_tiles(self, image):
        return [[image[y*self.tile_size:(y+1)*self.tile_size, x*self.tile_size:(x+1)*self.tile_size] for x in range(self.cols)] for y in range(self.rows)]

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
        return "Home"

    def count_crowns(self, tile):
        hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        lower_yellow = np.array([18, 90, 90])
        upper_yellow = np.array([40, 255, 255])
        mask = cv.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        crown_candidates = [cnt for cnt in contours if 70 < cv.contourArea(cnt) < 300]

        return len(crown_candidates)

    def find_areas(self, board):
        visited = [[False]*self.cols for _ in range(self.rows)]
        area_map = [[None]*self.cols for _ in range(self.rows)]
        area_scores = {}
        area_id = 0

        def dfs(r, c, terrain):
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return 0, 0
            if visited[r][c] or board[r][c][0] != terrain:
                return 0, 0
            visited[r][c] = True
            area_map[r][c] = area_id
            count = 1
            crowns = board[r][c][1]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                a, b = dfs(r+dr, c+dc, terrain)
                count += a
                crowns += b
            return count, crowns

        for r in range(self.rows):
            for c in range(self.cols):
                terrain = board[r][c][0]
                if not visited[r][c] and terrain not in ("Home", "Table"):
                    size, crowns = dfs(r, c, terrain)
                    if size > 0:
                        area_scores[area_id] = size * crowns
                        area_id += 1

        return area_map, area_scores

    def annotate_board(self, image, board, area_map, area_scores, total_score):
        font = cv.FONT_HERSHEY_SIMPLEX
        overlay = image.copy()
        area_colors = {area_id: (random.randint(60,255), random.randint(60,255), random.randint(60,255)) for area_id in set(filter(None.__ne__, sum(area_map, [])))}

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
                        cv.put
