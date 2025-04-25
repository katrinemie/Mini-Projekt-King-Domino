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

# Klasse der håndterer tile klassificering med SVM
class TileClassifierSVM:
    def __init__(self, input_folder, ground_truth_csv):
        self.input_folder = input_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        if not self.image_paths:
            raise FileNotFoundError("Ingen billeder fundet i mappen.")
        self.ground_truth = self.load_ground_truth(ground_truth_csv)
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_ground_truth(self, csv_path):
        df = pd.read_csv(csv_path)
        ground_truth = {}
        for img_id in df['image_id'].unique():
            matrix = [['' for _ in range(5)] for _ in range(5)]
            subset = df[df['image_id'] == img_id]
            for _, row in subset.iterrows():
                col = int(row['x'] // 100)
                r = int(row['y'] // 100)
                matrix[r][col] = row['terrain']
            ground_truth[f"{img_id}.jpg"] = matrix
        return ground_truth

    def extract_features(self, tile):
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        median_hsv = np.median(hsv.reshape(-1, 3), axis=0)
        return median_hsv

    def split_to_tiles(self, image):
        h, w = image.shape[:2]
        tile_h = h // 5
        tile_w = w // 5
        return [[image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w] for x in range(5)] for y in range(5)]

    def prepare_training_data(self):
        X = []
        y = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            if filename not in self.ground_truth:
                continue
            img = cv2.imread(path)
            tiles = self.split_to_tiles(img)
            true_labels = self.ground_truth[filename]
            for r in range(5):
                for c in range(5):
                    feature = self.extract_features(tiles[r][c])
                    label = true_labels[r][c]
                    if label != '':
                        X.append(feature)
                        y.append(label)
        return np.array(X), np.array(y)

    def train_svm(self):
        X, y = self.prepare_training_data()
        y_encoded = self.label_encoder.fit_transform(y)
        self.model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
        self.model.fit(X, y_encoded)
        print(f"SVM trænet på {len(X)} tiles.")

    def process_images(self):
        y_true = []
        y_pred = []

        for path in self.image_paths:
            filename = os.path.basename(path)
            if filename not in self.ground_truth:
                continue

            img = cv2.imread(path)
            tiles = self.split_to_tiles(img)
            true_labels = self.ground_truth[filename]

            print(f"\nBillede: {filename}")
            for r in range(5):
                for c in range(5):
                    feature = self.extract_features(tiles[r][c]).reshape(1, -1)
                    pred_encoded = self.model.predict(feature)[0]
                    pred = self.label_encoder.inverse_transform([pred_encoded])[0]
                    true = true_labels[r][c]

                    if true != '':
                        y_true.append(true)
                        y_pred.append(pred)
                        if pred != true:
                            print(f"Tile[{r},{c}] - Forventet: {true}, Fundet: {pred}")

        # Beregn confusion matrix og accuracy
        cm = confusion_matrix(y_true, y_pred, labels=self.label_encoder.classes_)
        acc = accuracy_score(y_true, y_pred) * 100

        print(f"\nSamlet Tile Classification Nøjagtighed med SVM: {acc:.2f}%")
        print("\nConfusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for Tile Classification')
        plt.show()

if __name__ == "__main__":
    print("Indlæser ground truth fra: ground_truth_train_split.csv")
    classifier = TileClassifierSVM(
        input_folder='splitted_dataset/train/cropped',
        ground_truth_csv='ground_truth_train_split.csv'
    )
    classifier.train_svm()
    classifier.process_images()
