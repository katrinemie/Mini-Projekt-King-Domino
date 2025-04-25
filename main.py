import os
import glob
import cv2 as cv
import numpy as np

from tile_classifier import TileClassifier
if __name__ == "__main__":
    classifier = TileClassifierSVM(
        input_folder='splitted_dataset/train/cropped',
        ground_truth_csv='ground_truth_train_split.csv'
    )
    classifier.train_svm()
    classifier.process_images()





from crown_detector import CrownDetector  # Importér klassen fra din fil (hvis den fx hedder crown_detector.py)

if __name__ == "__main__":
    detector = CrownDetector( ... )  # dine parametre her

    # Her modtager du værdierne fra process_images()
    tiles_with_crowns, total_tiles = detector.process_images()

    accuracy = (tiles_with_crowns / total_tiles) * 100 if total_tiles else 0
    print(f"\nSamlet Template Matching Nøjagtighed (Tiles med fundne kroner): {accuracy:.2f}%")

from score_calculator import ScoreCalculator

if __name__ == "__main__":
    calculator = ScoreCalculator()
    billede_nr = input("Indtast billednummer (uden .jpg): ")
    calculator.calculate_score(billede_nr)


from neighbour_detection import TileAnalyzer

if __name__ == "__main__":
    analyzer = TileAnalyzer(input_folder='splitted_dataset/train/cropped')
    analyzer.process_images()

