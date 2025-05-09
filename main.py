import cv2
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from crown_detecter import CrownDetector
from tile_classifier import TileClassifierSVM
from score_calculator import BoardScorer
from neighbour_detection import NeighbourDetection

# Funktion til at hente de sande etiketter fra en CSV-fil
def load_true_crown_labels_from_csv(label_file):
    try:
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
    except Exception as e:
        print(f"Error loading labels from CSV: {e}")
        return {}

# Funktion til at evaluere crown detection
def evaluate_crown_detection(true_crowns, predicted_crowns):
    all_true = []
    all_predicted = []
    for image_id in true_crowns:
        true_data = true_crowns[image_id]
        predicted_data = predicted_crowns.get(image_id, [])
        for (x, y, true_crowns_count), (px, py, predicted_crowns_count) in zip(true_data, predicted_data):
            all_true.append(true_crowns_count)
            all_predicted.append(predicted_crowns_count)
    
    precision = precision_score(all_true, all_predicted, average='macro', zero_division=0)
    recall = recall_score(all_true, all_predicted, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_predicted, average='macro', zero_division=0)
    accuracy = accuracy_score(all_true, all_predicted)
    return precision, recall, f1, accuracy, all_true, all_predicted

# Funktion til at plotte confusionmatrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.Blues

    cax = ax.matshow(cm, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title('Confusion Matrix for Crown Detection', fontsize=16, pad=20)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="black", fontsize=16)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    fig.colorbar(cax).remove()

    plt.tight_layout()
    plt.show()

#Funktion til at køre crown detecter test
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
        detected_crowns = crown_detector.detect_crowns(img, image_file)

        if detected_crowns is not None and np.any(detected_crowns):
            predicted_crowns[image_id] = []
            for row in range(5):
                for col in range(5):
                    count = detected_crowns[row, col]
                    predicted_crowns[image_id].append((col, row, count))
        else:
            print(f"No crowns detected in image {image_file}")

    if not true_crowns or not predicted_crowns:
        print("No valid data for crown detection evaluation.")
        return

    precision, recall, f1, acc, all_true, all_predicted = evaluate_crown_detection(true_crowns, predicted_crowns)

    print("\nCrown Detection Results:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")

    cm = confusion_matrix(all_true, all_predicted, labels=[1, 0])
    plot_confusion_matrix(cm, labels=["Crown", "No Crown"])

# Main funktionen til at køre alle tests
def main():
    image_path = r"splitted_dataset/test/cropped"
    label_file = r"ground_truth.csv"

    crown_templates = [
        r"crown_image/single_krone1.png",
        r"crown_image/single_krone2.png",
        r"crown_image/single_krone3.png",
        r"crown_image/single_krone4.png"
    ]
    crown_detector = CrownDetector(
        input_folder=image_path,
        template_paths=crown_templates,
        output_folder='output',
        scales=[0.8, 1.0, 1.2],
        angles=[0, 90, 180, 270],
        threshold=0.6
    )

    print("\nStarting Crown Detection Test")
    run_crown_detection_accuracy_test(image_path, crown_detector, label_file)

    print("\nStarting Tile Classifier Test")
    classifier = TileClassifierSVM('splitted_dataset/test/cropped', 'ground_truth.csv')
    classifier.train_svm()
    if classifier.model:
        classifier.evaluate()

    print("\nStarting Score Calculator Test")
    scorer = BoardScorer(
        input_folder="splitted_dataset/test/cropped",
        ground_truth_csv="ground_truth_scores.csv",
        output_folder="outputs"
    )
    scorer.run()

    print("\nStarting Neighbour Detection Test")
    neighbour_detector = NeighbourDetection('splitted_dataset/test/cropped')
    neighbour_detector.process_images()

if __name__ == "__main__":
    main()
