import os
import glob
import cv2 as cv
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

from tile_classifier import TileClassifierSVM
from neighbour_detection import TileAnalyzer  # Assuming CrownDetector has been replaced with TileAnalyzer


def load_ground_truth(csv_path):
    """Indlæser ground truth fra en CSV-fil og returnerer som dictionary."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f" Fejl: Ground truth CSV-fil ikke fundet på: {csv_path}")
        return {}
    except Exception as e:
        print(f" Fejl ved indlæsning af CSV {csv_path}: {e}")
        return {}

    ground_truth_data = {}
    for _, row in df.iterrows():
        filename = row['filename']
        r, c = int(row['row']), int(row['col'])
        label = row['true_label']

        if filename not in ground_truth_data:
            ground_truth_data[filename] = [['Unknown']*5 for _ in range(5)]
        ground_truth_data[filename][r][c] = label

    return ground_truth_data


def main_tile_classifier():
    print("Starter Tile Classifier med SVM...")

    classifier = TileClassifierSVM(
        input_folder='splitted_dataset/train/cropped',
        ground_truth_csv='ground_truth_train_split.csv'
    )

    classifier.train_svm()
    classifier.process_images()


def main_crown_detection():
    print("Starter Crown Detection med Template Matching...")

    # Initialiser TileAnalyzer (erstatte CrownDetector med TileAnalyzer)
    detector = TileAnalyzer(
        input_folder='splitted_dataset/train/cropped',
        template_paths=[
            'crown_templates/Skærmbillede 2025-04-23 kl. 13.08.35.png',
            'crown_templates/Skærmbillede 2025-04-23 kl. 13.08.47.png'
        ],
        output_folder='output',
        scales=[0.8, 1.0, 1.2],
        angles=[0, 90, 180, 270],
        threshold=0.6
    )

    # Indlæs ground truth fra CSV
    ground_truth = load_ground_truth('ground_truth_train_split.csv')

    # Kør detektion og evaluering
    detector.process_images(ground_truth)


def main_neighbour_detection():
    print("Starter Neighbor Detection...")

    # Initialiser TileAnalyzer (erstatte CrownDetector med TileAnalyzer)
    detector = TileAnalyzer(
        input_folder='splitted_dataset/train/cropped',
        template_paths=[
            'crown_image/opdateret_skærmbillede.png',
            'crown_image/opdateret_skærmbillede2.png',
            'crown_image/opdateret_skærmbillede3.png',
            'crown_image/opdateret_skærmbillede4.png'
        ],
        output_folder='outputs_with_crowns',
        scales=[0.9, 1.0, 1.2],
        angles=[0, 90, 180, 270],
        threshold=0.6
    )

    # Indlæs ground truth fra CSV
    ground_truth = load_ground_truth('ground_truth_train_split.csv')

    # Kør detektor
    detector.process_images(ground_truth)


if __name__ == "__main__":
    main_tile_classifier()  # Denne linje vælger hvilken main funktion du vil køre
    # main_crown_detection()
    # main_neighbour_detection()
