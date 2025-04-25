import cv2
import numpy as np
import os
import glob
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Klasse der håndterer tile klassificering med SVM
class TileClassifierSVM:
    def __init__(self, input_folder, ground_truth_csv):
        # Gemmer stien til input-billeder
        self.input_folder = input_folder
        # Finder alle .jpg filer i mappen
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        if not self.image_paths:
            # Fejl hvis ingen billeder findes
            raise FileNotFoundError("Ingen billeder fundet i mappen.")
         # Indlæser korrekt terræn fra CSV
        self.ground_truth = self.load_ground_truth(ground_truth_csv)
        # Placeholder til SVM modellen
        self.model = None
        # Initialiserer LabelEncoder til at håndtere labels
        self.label_encoder = LabelEncoder()
        
    # Indlæser ground truth data
    def load_ground_truth(self, csv_path):
        # Læser CSV-filen som dataframe
        df = pd.read_csv(csv_path)
        ground_truth = {}
        # Går igennem billede ID'er
        for img_id in df['image_id'].unique():
            # Opretter tom 5x5 matrix
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
        # Brug pipeline med scaler + SVM for bedre performance
        self.model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
        self.model.fit(X, y_encoded)
        print(f"SVM trænet på {len(X)} tiles.")

    def process_images(self):
        total_tiles = 0
        correct_tiles = 0

        for path in self.image_paths:
            filename = os.path.basename(path)
            if filename not in self.ground_truth:
                continue

            img = cv2.imread(path)
            tiles = self.split_to_tiles(img)
            true_labels = self.ground_truth[filename]

            print(f"\n Billede: {filename}")
            for r in range(5):
                for c in range(5):
                    feature = self.extract_features(tiles[r][c]).reshape(1, -1)
                    pred_encoded = self.model.predict(feature)[0]
                    pred = self.label_encoder.inverse_transform([pred_encoded])[0]
                    true = true_labels[r][c]

                    total_tiles += 1
                    if pred == true:
                        correct_tiles += 1
                    else:
                        print(f"Tile[{r},{c}] - Forventet: {true}, Fundet: {pred}")

        accuracy = (correct_tiles / total_tiles) * 100 if total_tiles else 0
        print(f"\n Samlet Tile Classification Nøjagtighed med SVM: {accuracy:.2f}%")

if __name__ == "__main__":
    classifier = TileClassifierSVM(
        input_folder='splitted_dataset/train/cropped',
        ground_truth_csv='ground_truth_split.csv'
    )
    classifier.train_svm()
    classifier.process_images()
