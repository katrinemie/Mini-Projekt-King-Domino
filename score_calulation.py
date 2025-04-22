import cv2 as cv
import numpy as np
import os
import random

tile_size = 100
rows, cols = 5, 5

def get_tiles(image):
    tiles = []
    for y in range(rows):
        tiles.append([])
        for x in range(cols):
            tiles[-1].append(image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size])
    return tiles

def get_terrain(tile):
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

def count_crowns(tile):
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)

    # Gule kroner (lys gul farve)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Fjern støj
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Find konturer
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    crown_candidates = [cnt for cnt in contours if 100 < cv.contourArea(cnt) < 800]

    return len(crown_candidates)

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
            if not visited[r][c] and terrain not in ("Home", "Table"):
                size, crowns, coords = dfs(r, c, terrain)
                if size > 0:
                    area_scores[area_id] = size * crowns
                    area_id += 1

    return area_map, area_scores

def annotate_board(image, board, area_map, area_scores, total_score):
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    overlay = image.copy()

    # Farver til hvert område
    area_colors = {}
    for area_id in set(filter(lambda x: x is not None, sum(area_map, []))):
        area_colors[area_id] = (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255)
        )

    # Debug-print i terminal
    print("\n=== Områder fundet ===")
    for area_id, score in area_scores.items():
        size = sum(row.count(area_id) for row in area_map)
        crowns = score // size if size > 0 else 0
        print(f"A#{area_id}: {size} felter, {crowns} kroner → {score}p")

    for r in range(rows):
        for c in range(cols):
            x, y = c * tile_size, r * tile_size
            terrain, crowns = board[r][c]
            area_id = area_map[r][c]

            # Tegn ramme rundt om område
            if area_id is not None:
                color = area_colors[area_id]
                cv.rectangle(overlay, (x, y), (x + tile_size, y + tile_size), color, 2)

            # Tekst på hver brik
            text_lines = []
            if terrain not in ("Home", "Table"):
                text_lines.append(f"{terrain}")
                text_lines.append(f"{crowns} crown(s)")
                if area_id is not None:
                    text_lines.append(f"A#{area_id} ({area_scores[area_id]}p)")

            for i, line in enumerate(text_lines):
                cv.putText(overlay, line, (x + 3, y + 15 + i * 15), font, scale, (0, 0, 0), thickness + 1, cv.LINE_AA)
                cv.putText(overlay, line, (x + 3, y + 15 + i * 15), font, scale, (255, 255, 255), thickness, cv.LINE_AA)

    # Total score nederst
    cv.rectangle(overlay, (0, tile_size * rows), (tile_size * cols, tile_size * rows + 30), (0, 0, 0), -1)
    cv.putText(overlay, f"Total score: {total_score}", (10, tile_size * rows + 22), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return overlay

# === HOVEDPROGRAM ===
billede_nr = input("Indtast billednummer (uden .jpg): ")
image_path = f"splitted_dataset/train/cropped/{billede_nr}.jpg"

if not os.path.isfile(image_path):
    print("Billedet blev ikke fundet.")
else:
    image = cv.imread(image_path)
    tiles = get_tiles(image)

    board = []
    for row in tiles:
        board_row = []
        for tile in row:
            terrain = get_terrain(tile)
            crowns = count_crowns(tile)
            board_row.append((terrain, crowns))
        board.append(board_row)

    area_map, area_scores = find_areas(board)
    total_score = sum(area_scores.values())

    annotated = annotate_board(image, board, area_map, area_scores, total_score)
    output_path = f"ground_truth_{billede_nr}.jpg"
    cv.imwrite(output_path, annotated)

    print(f"\n✔️ Ground truth-billedet er gemt som: {output_path}")
    print(f"✔️ Total score: {total_score} point")
