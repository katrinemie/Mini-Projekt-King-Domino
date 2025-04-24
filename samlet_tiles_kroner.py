import os
import cv2
import csv
import numpy as np
from collections import defaultdict

def load_ground_truth(csv_path):
    tiles_by_board = defaultdict(list)
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            x = int(row["x"])
            y = int(row["y"])
            terrain = row["terrain"]
            crowns = int(row["crowns"])
            tiles_by_board[image_id].append((x, y, terrain, crowns))
    return tiles_by_board

def assemble_boards(tile_folder, ground_truth, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_id, tiles in ground_truth.items():
        board_img = np.zeros((500, 500, 3), dtype=np.uint8)
        crown_total = 0

        for x, y, terrain, crowns in tiles:
            filename = f"tile_{image_id}_{x}_{y}.jpg"
            tile_path = os.path.join(tile_folder, filename)

            if not os.path.exists(tile_path):
                print(f"Mangler tile: {filename}")
                continue

            tile = cv2.imread(tile_path)
            if tile is None:
                print(f" Kunne ikke læse: {tile_path}")
                continue

            # Sæt tile på den rette placering
            board_img[y:y+100, x:x+100] = tile

            # --- Forbedret label-layout på hver tile ---
            label = f"{terrain} ({crowns})"
            label_position = (x + 5, y + 95)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Semi-transparent baggrund
            overlay = board_img.copy()
            cv2.rectangle(overlay, (x, y + 80), (x + 100, y + 100), (255, 255, 255), -1)
            alpha = 0.5
            board_img = cv2.addWeighted(overlay, alpha, board_img, 1 - alpha, 0)

            # Tekst
            cv2.putText(board_img, label, label_position, font, 0.4, (0, 0, 0), 1)

            crown_total += crowns

        # --- Diskret "Total kroner" i øverste hjørne ---
        total_text = f"Total kroner: {crown_total}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        overlay = board_img.copy()
        cv2.rectangle(overlay, (0, 0), (150, 20), (255, 255, 255), -1)
        alpha = 0.5
        board_img = cv2.addWeighted(overlay, alpha, board_img, 1 - alpha, 0)

        cv2.putText(board_img, total_text, (5, 15), font, 0.5, (0, 120, 0), 1)

        # Gem bræt
        out_path = os.path.join(output_folder, f"board_{image_id}.jpg")
        cv2.imwrite(out_path, board_img)
        print(f"Gemte: {out_path}")

# === Brug det her ===
csv_path = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\ground_truth_split.csv"
tile_folder = r"C:\Users\katri\Desktop\Kingkat\All_Tiles"
output_folder = r"C:\Users\katri\Desktop\Kingkat\Boards_annoteret"

ground_truth = load_ground_truth(csv_path)
assemble_boards(tile_folder, ground_truth, output_folder)
