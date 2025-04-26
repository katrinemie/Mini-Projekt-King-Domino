import cv2
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from crown_detecter import CrownDetector
from sklearn.metrics import confusion_matrix, accuracy_score
from tile_classifier import TileClassifierSVM
from score_calculator import BoardScorer
from neighbour_detection import NeighbourDetection

# Funktion til at indlæse de sande mærkater for kroner fra en CSV-fil
def load_true_crown_labels_from_csv(label_file):
    df = pd.read_csv(label_file)
    crowns_data = df[['image_id', 'x', 'y', 'crowns']]
    labels = {}
    
    for _, row in crowns_data.iterrows():
        image_id = row['image_id']
        x, y, crowns = row['x'], row['y'], row['crowns']
        
        if image_id not in labels:
            labels[image_id] = []
        
        labels[image_id].append((x, y, crowns))
    
    return labels

# Test funktion til at evaluere kronedetektionens præcision, recall, F1 score og nøjagtighed
def evaluate_crown_detection(true_crowns, predicted_crowns):
    all_true = []
    all_predicted = []

    for image_id in true_crowns:
        true_data = true_crowns[image_id]
        predicted_data = predicted_crowns.get(image_id, [])

        for (x, y, true_crowns_count), (_, _, predicted_crowns_count) in zip(true_data, predicted_data):
            all_true.append(true_crowns_count)
            all_predicted.append(predicted_crowns_count)

    precision = precision_score(all_true, all_predicted, average='macro', zero_division=0)
    recall = recall_score(all_true, all_predicted, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_predicted, average='macro', zero_division=0)
    accuracy = accuracy_score(all_true, all_predicted)

    return precision, recall, f1, accuracy

# Test for at køre kronedetektion og scoreevaluering på billederne
def run_crown_detection_accuracy_test(image_path, crown_detector, label_file):
    true_crowns = load_true_crown_labels_from_csv(label_file)
    predicted_crowns = {}

    for image_file in os.listdir(image_path):
        if not image_file.endswith(".jpg"):
            continue

        image_id = int(os.path.splitext(image_file)[0])
        img = cv2.imread(os.path.join(image_path, image_file))

        if img is None:
            print(f"Image {image_file} not found or unreadable.")
            continue

        # Kør kronedetektion på billedet
        hsv_tile = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        crown_count = len(crown_detector.detect_crowns(hsv_tile))

        predicted_crowns[image_id] = crown_count

    precision, recall, f1, accuracy_score = evaluate_crown_detection(true_crowns, predicted_crowns)

    print(f"\nCrown Detection Results:")
    print(f"Accuracy: {accuracy_score:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")

# Hovedfunktion til at køre alle tests og visualiseringer
def main():
    image_path = r"Cropped_and_corrected_boards"  # Stien til dine billeder
    label_file = r"ground_truth.csv"  # Stien til din ground truth CSV-fil

    # Initialiser kronedetektoren med dine skabelonbilleder
    crown_templates = [
        r"Reference_tiles\reference_crown_small1_rot90.JPG",
        r"Reference_tiles\reference_crown_small1_rot180.JPG",
        r"Reference_tiles\reference_crown_small1_rot270.JPG",
        r"Reference_tiles\reference_crown_small1.jpg"
    ]
    crown_detector = CrownDetector(crown_templates)

    # Initialiser din classifier (her bruger vi din TileClassifierSVM)
    classifier = TileClassifierSVM()
    classifier.run_pipeline(image_path)  # Kør din pipeline, hvis nødvendigt

    # Kør test for kronedetektionens nøjagtighed
    run_crown_detection_accuracy_test(image_path, crown_detector, label_file)

if __name__ == "__main__":
    main()
