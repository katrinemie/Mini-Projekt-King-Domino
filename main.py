import cv2
import numpy as np
import os
import glob
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


from tile_classifier import TileClassifierSVM
from neighbour_detection import NeighbourDetector


if __name__ == "__main__":
    print("Indlæser ground truth fra: ground_truth_train_split.csv")
    
    # Initialiserer TileClassifier med input og ground truth fil
    classifier = TileClassifierSVM(
        input_folder='splitted_dataset/test/cropped',  # Mappen med billeder
        ground_truth_csv='ground_truth_test_split.csv'  # Mappen med ground truth data (CSV)
    )
    
    # Træn SVM modellen
    classifier.train_svm()
    
    # Test og evaluer modellen på billederne
    classifier.process_images()

if __name__ == "__main__":
    # Angiv input-mappen og ground truth CSV-fil til testdata
    INPUT_FOLDER = 'splitted_dataset/test/cropped'  # Mappen med testbilleder
    GROUND_TRUTH_CSV = 'ground_truth_test_split.csv'  # CSV-fil med ground truth data for testdata

    # Hvis du ikke har ground truth-data til test, kan du lade den være None:
    # GROUND_TRUTH_CSV = None

    # Initialiser NeighbourDetector med input-mappe og ground truth CSV
    detector = NeighbourDetector(INPUT_FOLDER, GROUND_TRUTH_CSV)

    # Processér billederne i testdataene
    detector.process_images()
