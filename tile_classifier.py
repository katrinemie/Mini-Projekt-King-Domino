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
from typing import List, Dict, Tuple, Optional, Any


class TileClassifierSVM:
    # Håndtering af klassificering af billeder baseret på gridopdeling
    def __init__(self, input_folder: str, ground_truth_csv: str, grid_size: int = 5):
        self.input_folder = input_folder
        self.ground_truth_csv = ground_truth_csv
        self.grid_size = grid_size
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        self.ground_truth = self.load_ground_truth(ground_truth_csv) if os.path.exists(ground_truth_csv) else {}
        self.model: Optional[Any] = None
        self.label_encoder = LabelEncoder()

     # Indlæser ground truth data fra en CSV-fil og opretter en dictionary med terrain labels for hvert billede
    def load_ground_truth(self, csv_path: str) -> Dict[str, List[List[str]]]:
        if not os.path.exists(csv_path):
            return {}
        df = pd.read_csv(csv_path)
        if not {'image_id', 'x', 'y', 'terrain'}.issubset(df.columns):
            return {}
        
        ground_truth: Dict[str, List[List[str]]] = {}
        h, w = cv2.imread(self.image_paths[0]).shape[:2] if self.image_paths else (0, 0)
        tile_h, tile_w = h // self.grid_size, w // self.grid_size
        if tile_w == 0 or tile_h == 0:
            return ground_truth
        
        for img_id in df['image_id'].unique():
            filename = f"{str(img_id)}.jpg"
            matrix = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            
            for _, row in df[df['image_id'] == img_id].iterrows():
                col, row_idx = int(row['x'] // tile_w), int(row['y'] // tile_h)
                if 0 <= row_idx < self.grid_size and 0 <= col < self.grid_size:
                    terrain = row['terrain']
                    matrix[row_idx][col] = str(terrain) if pd.notna(terrain) else ''
            ground_truth[filename] = matrix

        return ground_truth
    # Ekstraherer farvebaserede features fra en given tile
    def extract_features(self, tile: np.ndarray) -> Optional[np.ndarray]:
        if tile is None or tile.size == 0:
            return None
        
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        median_hsv = np.median(hsv.reshape(-1, 3), axis=0)
        if np.any(np.isnan(median_hsv)) or np.any(np.isinf(median_hsv)):
            return None
        return median_hsv

    # Opdeler et billede i mindre tiles baseret på grid-størrelsen
    def split_to_tiles(self, image: np.ndarray) -> List[List[Optional[np.ndarray]]]:
        if image is None:
            return [[None] * self.grid_size for _ in range(self.grid_size)]
        h, w = image.shape[:2]
        tile_h, tile_w = h // self.grid_size, w // self.grid_size
        return [[image[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] if image[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w].size > 0 else None
                 for c in range(self.grid_size)] for r in range(self.grid_size)]
    
    # Forbereder træningsdata ved at udtrække features fra billederne og matche dem med deres labels
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for path in [p for p in self.image_paths if os.path.basename(p) in self.ground_truth]:
            img = cv2.imread(path)
            if img is None:
                continue
            tiles = self.split_to_tiles(img)
            true_labels = self.ground_truth[os.path.basename(path)]
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    tile, label = tiles[r][c], true_labels[r][c]
                    if isinstance(label, str) and label and tile is not None:
                        feature = self.extract_features(tile)
                        if feature is not None:
                            X.append(feature)
                            y.append(label)
        return np.array(X), np.array(y)

     # Træner en Support Vector Machine (SVM) model med de forberedte træningsdata
    def train_svm(self, C: float = 10.0, kernel: str = 'rbf', gamma: str = 'scale') -> None:
        X, y = self.prepare_training_data()
        if X.shape[0] < 2:
            return
        y_encoded = self.label_encoder.fit_transform(y)
        self.model = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=42)
        ).fit(X, y_encoded)

    # Forudsiger labelen for en given tile baseret på den trænede SVM-model
    def predict_tile(self, tile: np.ndarray) -> Optional[str]:
        if self.model is None or tile is None:
            return None
        feature = self.extract_features(tile)
        if feature is None:
            return None
        prediction = self.model.predict(feature.reshape(1, -1))[0]
        return self.label_encoder.inverse_transform([prediction])[0]

    # Evaluerer modellen ved at sammenligne forudsigelser med de sande labels
    def evaluate(self, eval_folder: Optional[str] = None, eval_csv: Optional[str] = None) -> None:
        if self.model is None:
            return

        y_true, y_pred = [], []
        target_folder = eval_folder or self.input_folder
        target_csv = eval_csv or self.ground_truth_csv
        eval_ground_truth = (
            self.ground_truth if target_folder == self.input_folder and target_csv == self.ground_truth_csv
            else self.load_ground_truth(target_csv)
        )

        for path in [p for p in glob.glob(os.path.join(target_folder, '*.jpg')) if os.path.basename(p) in eval_ground_truth]:
            img = cv2.imread(path)
            if img is None:
                continue
            tiles = self.split_to_tiles(img)
            true_labels = eval_ground_truth[os.path.basename(path)]
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    tile, label = tiles[r][c], true_labels[r][c]
                    if isinstance(label, str) and label and tile is not None:
                        pred = self.predict_tile(tile)
                        if pred:
                            y_true.append(label)
                            y_pred.append(pred)

        if not y_true:
            return

        cm_labels = list(self.label_encoder.classes_)
        acc = accuracy_score(y_true, y_pred) * 100
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        print(f"\nNøjagtighed: {acc:.2f}%")
        print("\nConfusion Matrix:\n", pd.DataFrame(cm, index=cm_labels, columns=cm_labels))

        plt.figure(figsize=(10, 7))
        sns.heatmap(pd.DataFrame(cm, index=cm_labels, columns=cm_labels), annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Accuracy: {acc:.2f}%)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    TRAIN_IMAGE_FOLDER = 'splitted_dataset/train/cropped'
    TRAIN_GROUND_TRUTH_CSV = 'ground_truth.csv'
    classifier = TileClassifierSVM(TRAIN_IMAGE_FOLDER, TRAIN_GROUND_TRUTH_CSV)
    classifier.train_svm()
    if classifier.model:
        classifier.evaluate()
