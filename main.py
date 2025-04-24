import os
import glob
import cv2 as cv
import numpy as np

from tile_classifier import TileClassifier

if __name__ == "__main__":
    classifier = TileClassifier(input_folder='splitted_dataset/train/cropped')
    classifier.process_images()

from crown_detector import CrownDetector  # Importér klassen fra din fil (hvis den fx hedder crown_detector.py)

if __name__ == "__main__":
    detector = CrownDetector(
        input_folder='splitted_dataset/train/cropped',
        template_paths=[
            'opdateret_skærmbillede.png',
            'opdateret_skærmbillede2.png',
            'opdateret_skærmbillede3.png',
            'opdateret_skærmbillede4.png'
        ],
        output_folder='outputs_with_crowns',
        scales=[0.9, 1.0, 1.2],
        angles=[0, 90, 180, 270],
        threshold=0.6
    )

    detector.process_images()
from score_calculator import ScoreCalculator

if __name__ == "__main__":
    calculator = ScoreCalculator()
    billede_nr = input("Indtast billednummer (uden .jpg): ")
    calculator.calculate_score(billede_nr)


from neighbour_detection import TileAnalyzer

if __name__ == "__main__":
    analyzer = TileAnalyzer(input_folder='splitted_dataset/train/cropped')
    analyzer.process_images()
