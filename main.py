import cv2
import numpy as np
import os
import glob
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, classification_report

class NeighbourDetector:
    def __init__(self, input_folder, ground_truth_folder=None):
        self.input_folder = input_folder
        self.ground_truth_folder = ground_truth_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        
        if not self.image_paths:
            raise FileNotFoundError("Ingen billeder fundet i input-mappen.")
        
        # Terræntype farver og klasser
        self.terrain_colors = {
            "Field": (255, 215, 0),    # Guld
            "Forest": (34, 139, 34),   # Skovgrøn
            "Lake": (65, 105, 225),    # Kongeblå
            "Grassland": (152, 251, 152),  # Lysegrøn
            "Swamp": (139, 137, 112),   # Sumpskimmel
            "Mine": (169, 169, 169),    # Mørkegrå
            "Home": (255, 99, 71),     # Tomat
            "Unknown": (220, 220, 220)  # Lysgrå
        }
        self.classes = list(self.terrain_colors.keys())
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Op, ned, venstre, højre

    def classify_tile(self, tile):
        """Klassificer en tile baseret på HSV-farveværdier"""
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
        """Opdel billede i 5x5 grid af tiles"""
        h, w = image.shape[:2]
        return [np.hsplit(row, 5) for row in np.vsplit(image, 5)]

    def find_neighbours(self, labels):
        """Identificer naboer af samme type for hver tile"""
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
        """Visualiser resultater med 3 underplots"""
        plt.figure(figsize=(14, 10))
        
        # Terræntyper
        ax1 = plt.subplot(1, 3, 1)
        terrain_map = np.zeros((5, 5, 3), dtype=np.uint8)
        for r in range(5):
            for c in range(5):
                terrain_map[r, c] = self.terrain_colors.get(labels[r][c], (220, 220, 220))
        plt.imshow(terrain_map)
        plt.title('Terræntyper')
        
        # Originalbillede
        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Originalbillede')
        
        # Naboanalyse
        ax3 = plt.subplot(1, 3, 3)
        h, w = image.shape[:2]
        tile_h, tile_w = h // 5, w // 5
        
        for r in range(5):
            for c in range(5):
                color = np.array(self.terrain_colors[labels[r][c]])/255
                rect = plt.Rectangle((c*tile_w, r*tile_h), tile_w, tile_h, 
                                   facecolor=color, edgecolor='white', alpha=0.7)
                ax3.add_patch(rect)
                ax3.text(c*tile_w + tile_w/2, r*tile_h + tile_h/2, 
                        str(neighbour_counts[r, c]),
                        ha='center', va='center', fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        for (r1, c1), (r2, c2) in connections:
            x1, y1 = c1*tile_w + tile_w/2, r1*tile_h + tile_h/2
            x2, y2 = c2*tile_w + tile_w/2, r2*tile_h + tile_h/2
            ax3.plot([x1, x2], [y1, y2], 'white', linewidth=1.5, alpha=0.7)
        
        ax3.set_title('Naboer (tal = antal naboer)')
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()

    def evaluate_performance(self, all_true, all_pred):
        """Evaluer modelperformance med confusion matrix og klassifikationsrapport"""
        # Beregn nøjagtighed
        accuracy = np.mean(np.array(all_true) == np.array(all_pred))
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(all_true, all_pred, labels=self.classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix (Antal tiles)')
        plt.xlabel('Forudsagt')
        plt.ylabel('Faktisk')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Detaljeret rapport
        print("\nDetaljeret klassifikationsrapport:")
        print(classification_report(all_true, all_pred, target_names=self.classes, zero_division=0))

    def process_images(self):
        """Hovedprocessering af billeder med evaluering"""
        all_true, all_pred = [], []
        
        for path in self.image_paths:
            print(f"\nBehandler {os.path.basename(path)}...")
            img = cv2.imread(path)
            if img is None:
                print(f"Kunne ikke læse {path}")
                continue

            # Klassificer tiles
            tiles = self.split_to_tiles(img)
            labels = [[self.classify_tile(tile) for tile in row] for row in tiles]
            
            # Find naboer
            connections, neighbour_counts = self.find_neighbours(labels)
            
            # Hvis ground truth findes
            if self.ground_truth_folder:
                gt_path = os.path.join(self.ground_truth_folder, os.path.basename(path))
                if os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path)
                    if gt_img is not None:
                        gt_tiles = self.split_to_tiles(gt_img)
                        gt_labels = [[self.classify_tile(tile) for tile in row] for row in gt_tiles]
                        all_true.extend([gt_labels[r][c] for r in range(5) for c in range(5)])
                        all_pred.extend([labels[r][c] for r in range(5) for c in range(5)])

            # Visualisering
            self.visualize_results(img, labels, connections, neighbour_counts)
            
            # Udskriv detaljer
            print("\nTile detaljer:")
            for r in range(5):
                for c in range(5):
                    print(f"({r},{c}): {labels[r][c]} ({neighbour_counts[r, c]} naboer)")

        # Evaluer hvis vi har ground truth
        if all_true and all_pred:
            self.evaluate_performance(all_true, all_pred)
        elif self.ground_truth_folder:
            print("\n⚠️ Ingen ground truth data fundet til evaluering")

if __name__ == "__main__":
    # Konfiguration - RET DISSE STIER!
    INPUT_FOLDER = 'splitted_dataset/train/cropped'  # Mappe med inputbilleder
    GROUND_TRUTH_FOLDER = None  # Sæt til mappe med ground truth billeder hvis tilgængelige
    
    print("Starter naboanalyse...")
    try:
        detector = NeighbourDetector(INPUT_FOLDER, GROUND_TRUTH_FOLDER)
        detector.process_images()
    except Exception as e:
        print(f"Fejl: {e}")