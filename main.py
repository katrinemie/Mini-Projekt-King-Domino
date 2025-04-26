import cv2
import numpy as np
import os
import csv
from tile_classifier import TileClassifierSVM
from neighbour_detection import NeighbourDetector
from crown_detecter import CrownDetector

class ScoreCalculator:
    def calculate_score(board, crowns):
        """Beregn score baseret på board (terrain) og crowns (kroner)."""
        visited = [[False]*len(board[0]) for _ in range(len(board))]
        total_score = 0

        def dfs(r, c, terrain):
            if (r < 0 or r >= len(board) or c < 0 or c >= len(board[0]) or 
                visited[r][c] or board[r][c] != terrain):
                return 0, 0
            visited[r][c] = True
            count = 1
            current_crowns = crowns[r][c]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                cnt, crn = dfs(r+dr, c+dc, terrain)
                count += cnt
                current_crowns += crn
            return count, current_crowns

        for r in range(len(board)):
            for c in range(len(board[0])):
                terrain = board[r][c]
                if not visited[r][c] and terrain != "Home":
                    size, region_crowns = dfs(r, c, terrain)
                    if region_crowns > 0:
                        total_score += size * region_crowns
        return total_score

def test_score_calculator():
    print("\n=== 🧪 TESTING SCORE CALCULATOR ===\n")
    
    # Test Case 1: Simpel 3x3 board med én region
    board1 = [
        ["Forest", "Forest", "Grass"],
        ["Forest", "Grass", "Grass"],
        ["Grass", "Grass", "Grass"]
    ]
    crowns1 = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    expected1 = 3 * 1  # 3 tiles * 1 crown
    result1 = ScoreCalculator.calculate_score(board1, crowns1)
    print(f"Test 1: {'✅' if result1 == expected1 else '❌'} Expected {expected1}, Got {result1}")

    # Test Case 2: Flere regioner med kroner
    board2 = [
        ["Forest", "Water", "Grass"],
        ["Forest", "Grass", "Grass"],
        ["Grass", "Grass", "Water"]
    ]
    crowns2 = [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ]
    expected2 = (2*1) + (4*2) + (1*1)  # 2 + 8 + 1 = 11
    result2 = ScoreCalculator.calculate_score(board2, crowns2)
    print(f"Test 2: {'✅' if result2 == expected2 else '❌'} Expected {expected2}, Got {result2}")

    # Test Case 3: Ingen kroner = 0 point
    board3 = [["Forest" for _ in range(3)] for _ in range(3)]
    crowns3 = [[0 for _ in range(3)] for _ in range(3)]
    expected3 = 0
    result3 = ScoreCalculator.calculate_score(board3, crowns3)
    print(f"Test 3: {'✅' if result3 == expected3 else '❌'} Expected {expected3}, Got {result3}")

if __name__ == "__main__":
    try:
        # === 1. Kør enhedstests ===
        test_score_calculator()

        # === 2. Valider mod ground truth fra CSV ===
        print("\n=== 🔍 VALIDERING MOD GROUND TRUTH ===")
        ground_truth = {}
        with open('ground_truth_scores.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                ground_truth[row['Image']] = int(row['Score'])

        input_folder = 'splitted_dataset/test/cropped'
        score_data = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                # Simuler board og crowns fra dit system
                board = [["Forest" for _ in range(5)] for _ in range(5)]  # TODO: Erstat med din egen logik
                crowns = [[0 for _ in range(5)] for _ in range(5)]       # TODO: Erstat med CrownDetector-output
                pred_score = ScoreCalculator.calculate_score(board, crowns)
                true_score = ground_truth.get(filename, None)
                
                if true_score is not None:
                    error = abs(pred_score - true_score)
                    score_data.append((filename, error))
                    print(f"📊 {filename}: Predicted={pred_score}, Ground Truth={true_score}, Error={error}")

        if score_data:
            mae = sum(error for _, error in score_data) / len(score_data)
            print(f"\n🔍 Gennemsnitlig fejl (MAE): {mae:.2f}")

    except Exception as e:
        print(f"❌ Kritisk fejl: {e}")
