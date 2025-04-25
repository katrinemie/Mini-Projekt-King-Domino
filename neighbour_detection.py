import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd

class NeighbourDetector:
    def __init__(self, input_folder, ground_truth_csv=None):
        self.input_folder = input_folder
        self.ground_truth_csv = ground_truth_csv
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        
        if not self.image_paths:
            raise FileNotFoundError("Ingen billeder fundet i input-mappen.")
        
        # Terræntype farver og klasser
        self.terrain_colors = {
            "Field": (255, 215, 0),
            "Forest": (34, 139, 34),
            "Lake": (65, 105, 225),
            "Grassland": (152, 251, 152),
            "Swamp": (139, 137, 112),
            "Mine": (169, 169, 169),
            "Home": (255, 99, 71),
            "Unknown": (220, 220, 220)
        }
        self.classes = list(self.terrain_colors.keys())
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Indlæs ground truth hvis CSV eksisterer
        self.gt_data = None
        if ground_truth_csv and os.path.exists(ground_truth_csv):
            self.gt_data = pd.read_csv(ground_truth_csv)

    def classify_tile(self, tile):
        """Klassificer en tile baseret på farve"""
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        hue, sat, val = np.median(hsv.reshape(-1, 3), axis=0)

        if 21.5 < hue < 27.5 and 225 < sat < 255 and 104 < val < 210:
            return "Field"
        if 25 < hue < 60 and 88 < sat < 247 and 24 < val < 78:
            return "Forest"
        if 90 < hue < 130 and 100 < sat < 255 and 100 < val < 230:
            return "Lake"
        if 34 < hue < 46 and 150 < sat < 255 and 90 < val < 180:
            return "Grassland"
        if 16 < hue < 27 and 66 < sat < 180 and 75 < val < 140:
            return "Swamp"
        if 19 < hue < 27 and 39 < sat < 150 and 29 < val < 80:
            return "Mine"
        if sat < 60 and 60 < val < 200:
            return "Home"
        return "Unknown"

    def split_to_tiles(self, image):
        """Opdel billede i 5x5 grid"""
        h, w = image.shape[:2]
        return [np.hsplit(row, 5) for row in np.vsplit(image, 5)]

    def find_neighbours(self, labels):
        """Find naboer af samme type"""
        connections = []
        neighbour_counts = np.zeros((5, 5), dtype=int)
        
        for r in range(5):
            for c in range(5):
                current_type = labels[r][c]
                for dr, dc in self.directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and labels[nr][nc] == current_type:
                        neighbour_counts[r, c] += 1
                        connections.append(((r, c), (nr, nc)))
        return connections, neighbour_counts

    def visualize_results(self, image, labels, connections, neighbour_counts):
        """Visualiser naboanalyse"""
        plt.figure(figsize=(14, 6))
        
        # Originalbillede
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Originalbillede')
        plt.axis('off')
        
        # Naboanalyse
        plt.subplot(1, 2, 2)
        h, w = image.shape[:2]
        tile_h, tile_w = h // 5, w // 5
        
        for r in range(5):
            for c in range(5):
                color = np.array(self.terrain_colors[labels[r][c]])/255
                rect = plt.Rectangle((c*tile_w, r*tile_h), tile_w, tile_h, 
                                    facecolor=color, edgecolor='white', alpha=0.7)
                plt.gca().add_patch(rect)
                plt.text(c*tile_w + tile_w/2, r*tile_h + tile_h/2,
                        str(neighbour_counts[r, c]),
                        ha='center', va='center', fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        for (r1, c1), (r2, c2) in connections:
            plt.plot([c1*tile_w + tile_w/2, c2*tile_w + tile_w/2],
                    [r1*tile_h + tile_h/2, r2*tile_h + tile_h/2],
                    'white', linewidth=1.5, alpha=0.7)
        
        plt.title('Naboanalyse (Antal naboer)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def evaluate_performance(self, all_true, all_pred):
        """Vis evalueringsmetrics"""
        # Accuracy
        accuracy = accuracy_score(all_true, all_pred)
        print(f"\nAccuracy: {accuracy:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(all_true, all_pred, labels=self.classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Forudsagt')
        plt.ylabel('Faktisk')
        plt.xticks(rotation=45)
        plt.show()
        
        # Klassifikationsrapport
        print("\nKlassifikationsrapport:")
        print(classification_report(all_true, all_pred, target_names=self.classes, zero_division=0))

    def process_images(self):
        """Hovedprocessen"""
        all_true, all_pred = [], []
        
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            # Klassificer tiles
            tiles = self.split_to_tiles(img)
            labels = [[self.classify_tile(tile) for tile in row] for row in tiles]
            
            # Find naboer
            connections, neighbour_counts = self.find_neighbours(labels)
            
            # Vis resultater
            self.visualize_results(img, labels, connections, neighbour_counts)
            
            # Hvis ground truth findes
            if self.gt_data is not None:
                filename = os.path.basename(path)
                gt_labels = self.gt_data[self.gt_data['image_name'] == filename]
                
                if not gt_labels.empty:
                    for _, row in gt_labels.iterrows():
                        r, c = row['row'], row['col']
                        all_true.append(row['terrain_type'])
                        all_pred.append(labels[r][c])

        # Evaluer hvis vi har data
        if all_true:
            self.evaluate_performance(all_true, all_pred)
        else:
            print("\nIngen ground truth til evaluering:")
            self.demo_evaluation()

    def demo_evaluation(self):
        """Simuler evaluering med tilfældige data"""
        all_true = np.random.choice(self.classes, size=100)
        all_pred = np.random.choice(self.classes, size=100)
        self.evaluate_performance(all_true, all_pred)
        print("\nNOTE: Dette er simulerede data! For reel evaluering, tilføj ground truth.")

if __name__ == "__main__":
    INPUT_FOLDER = 'splitted_dataset/train/cropped'  
    GROUND_TRUTH_CSV = 'ground_truth_train_split.csv'  
    detector = NeighbourDetector(INPUT_FOLDER, GROUND_TRUTH_CSV)
    detector.process_images()