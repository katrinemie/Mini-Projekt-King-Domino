import os
from crown_detecter import CrownDetector, ground_truth_from_csv
from neighbour_detection import NeighbourDetector
from score_calculator import compare_with_ground_truth, save_score_csv
from tile_classifier import TileClassifierSVM
import cv2

# === Mapper og filer ===
CROPPED_FOLDER = 'splitted_dataset/test/cropped'
GROUND_TRUTH_CSV = 'ground_truth.csv'
GROUND_TRUTH_SCORES_CSV = 'ground_truth_scores.csv'
SCORE_OUTPUT_FOLDER = 'outputs'
os.makedirs(SCORE_OUTPUT_FOLDER, exist_ok=True)

def run_crown_detector():
    print("\n=== Crown Detection ===")
    ground_truth = ground_truth_from_csv(GROUND_TRUTH_CSV)
    detector = CrownDetector(
        input_folder=CROPPED_FOLDER,
        template_paths=[
            'board_templates/Skærmbillede 2025-04-23 kl. 13.08.35.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.08.47.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.08.56.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.09.05.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.09.13.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.09.25.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.09.44.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.09.51.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.10.19.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.11.34.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.12.43.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.13.03.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.13.14.png',
            'board_templates/Skærmbillede 2025-04-23 kl. 13.13.32.png'
        ],
        output_folder='output',
        scales=[0.8, 1.0, 1.2],
        angles=[0, 90, 180, 270],
        threshold=0.6
    )
    detector.process_images(ground_truth)

def run_neighbour_detection():
    print("\n=== Neighbour Detection ===")
    detector = NeighbourDetector(CROPPED_FOLDER)
    detector.process_images()

def run_score_calculator():
    print("\n=== Score Calculation ===")
    from score_calculator import get_tiles, get_terrain, count_crowns, find_areas, annotate_board

    score_data = []

    for filename in sorted(os.listdir(CROPPED_FOLDER)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(CROPPED_FOLDER, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Kunne ikke indlæse: {filename}")
                continue

            tiles = get_tiles(image)
            board = [[(get_terrain(tile), count_crowns(tile)) for tile in row] for row in tiles]
            area_map, area_scores = find_areas(board)
            total_score = sum(area_scores.values())
            annotated = annotate_board(image, board, area_map, area_scores, total_score, filename)

            output_img_path = os.path.join(SCORE_OUTPUT_FOLDER, f"score_calculator_output_{filename}")
            cv2.imwrite(output_img_path, annotated)
            score_data.append((filename, total_score))
            print(f"✔ Gemte: {output_img_path} | Score: {total_score}")

    save_score_csv(score_data)
    compare_with_ground_truth(score_data, GROUND_TRUTH_SCORES_CSV)

def run_tile_classifier():
    print("\n=== Tile Classifier (SVM) ===")
    classifier = TileClassifierSVM(CROPPED_FOLDER, GROUND_TRUTH_CSV)
    classifier.train_svm()
    if classifier.model:
        classifier.evaluate()

# === Main Call ===
if __name__ == "__main__":
    run_crown_detector()
    run_neighbour_detection()
    run_score_calculator()
    run_tile_classifier()
