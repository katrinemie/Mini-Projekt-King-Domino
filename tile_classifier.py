import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

class NeighbourDetector:
    def __init__(self, input_folder, classifier):
        """
        Initialiser neighbour detector med en tile classifier
        
        Args:
            input_folder: Mappe med inputbilleder
            classifier: Initialiseret tile classifier (TileClassifierSVM instans)
        """
        self.input_folder = input_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        self.classifier = classifier
        
        if not self.image_paths:
            raise FileNotFoundError("Ingen billeder fundet i input-mappen.")
        
        # Terræntype farver og klasser (skal matche din SVM classifier)
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

    def classify_tile(self, tile):
        """Bruger SVM classifier til at klassificere en tile"""
        # Ekstraher features på samme måde som i din SVM classifier
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        feature = np.median(hsv.reshape(-1, 3), axis=0).reshape(1, -1)
        
        # Brug SVM til at forudsige
        pred_encoded = self.classifier.model.predict(feature)[0]
        return self.classifier.label_encoder.inverse_transform([pred_encoded])[0]

    def split_to_tiles(self, image):
        """Opdel billede i 5x5 grid - samme implementering som i din SVM classifier"""
        h, w = image.shape[:2]
        tile_h = h // 5
        tile_w = w // 5
        return [[image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w] for x in range(5)] for y in range(5)]

    def find_neighbours(self, labels):
        """
        Find naboer af samme type med avanceret analyse
        
        Returnerer:
            connections: Liste af nabo-forbindelser
            neighbour_counts: Matrix med antal naboer per tile
            regions: Dictionary med regioninformation
        """
        connections = []
        neighbour_counts = np.zeros((5, 5), dtype=int)
        region_map = -np.ones((5, 5), dtype=int)
        current_region = 0
        regions = {}
        
        # Find sammenhængende regioner
        for r in range(5):
            for c in range(5):
                if region_map[r, c] == -1:  # Ikke tildelt til region endnu
                    region_type = labels[r][c]
                    region_tiles = self._flood_fill(labels, region_map, r, c, current_region)
                    regions[current_region] = {
                        'type': region_type,
                        'tiles': region_tiles,
                        'size': len(region_tiles)
                    }
                    current_region += 1
        
        # Beregn naboer og forbindelser
        for r in range(5):
            for c in range(5):
                current_type = labels[r][c]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Op, ned, venstre, højre
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and labels[nr][nc] == current_type:
                        neighbour_counts[r, c] += 1
                        if (r, c) < (nr, nc):  # Undgå duplikerede forbindelser
                            connections.append(((r, c), (nr, nc)))
        
        return connections, neighbour_counts, regions

    def _flood_fill(self, labels, region_map, r, c, region_id):
        """Hjælpefunktion til at finde sammenhængende regioner"""
        stack = [(r, c)]
        region_type = labels[r][c]
        region_tiles = []
        
        while stack:
            x, y = stack.pop()
            if region_map[x, y] != -1:
                continue
                
            region_map[x, y] = region_id
            region_tiles.append((x, y))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dr, y + dc
                if 0 <= nx < 5 and 0 <= ny < 5 and labels[nx][ny] == region_type and region_map[nx, ny] == -1:
                    stack.append((nx, ny))
        
        return region_tiles

    def visualize_results(self, image, labels, connections, neighbour_counts, regions):
        """Avanceret visualisering med regioninformation"""
        plt.figure(figsize=(16, 6))
        
        # Originalbillede
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Originalbillede')
        plt.axis('off')
        
        # Naboanalyse visualisering
        plt.subplot(1, 2, 2)
        h, w = image.shape[:2]
        tile_h, tile_w = h // 5, w // 5
        
        # Tegn tiles med farver
        for r in range(5):
            for c in range(5):
                color = np.array(self.terrain_colors.get(labels[r][c], (220, 220, 220)))/255
                rect = plt.Rectangle((c*tile_w, r*tile_h), tile_w, tile_h, 
                                    facecolor=color, edgecolor='white', alpha=0.7)
                plt.gca().add_patch(rect)
                
                # Vis antal naboer og region-id
                region_id = regions.get(region_map[r, c], {}).get('size', '')
                plt.text(c*tile_w + tile_w/2, r*tile_h + tile_h/2,
                        f"{neighbour_counts[r, c]}\n(R:{region_id})",
                        ha='center', va='center', fontsize=8, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        # Tegn linjer mellem forbundne naboer
        for (r1, c1), (r2, c2) in connections:
            plt.plot([c1*tile_w + tile_w/2, c2*tile_w + tile_w/2],
                    [r1*tile_h + tile_h/2, r2*tile_h + tile_h/2],
                    'white', linewidth=1.5, alpha=0.7)
        
        # Legend
        terrain_types = sorted(list(set(labels[r][c] for r in range(5) for c in range(5))))
        legend_elements = [Patch(facecolor=np.array(self.terrain_colors.get(name, (220, 220, 220)))/255, 
                          label=name) for name in terrain_types]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Naboanalyse (Antal naboer og regionstørrelse)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Vis regionstatistik
        print("\nRegionstatistik:")
        for region_id, data in regions.items():
            print(f"Region {region_id}: Type={data['type']}, Størrelse={data['size']}")

    def process_images(self):
        """Hovedprocessen der behandler alle billeder"""
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Kunne ikke indlæse billede: {path}")
                continue

            # Opdel i tiles og klassificer
            tiles = self.split_to_tiles(img)
            labels = [[self.classify_tile(tile) for tile in row] for row in tiles]
            
            # Find naboer og regioner
            connections, neighbour_counts, regions = self.find_neighbours(labels)
            
            # Vis resultater
            print(f"\nAnalyserer {os.path.basename(path)}:")
            self.visualize_results(img, labels, connections, neighbour_counts, regions)

if __name__ == "__main__":
    # Initialiser din SVM classifier først
    print("Initialiserer tile classifier...")
    svm_classifier = TileClassifierSVM(
        input_folder='splitted_dataset/train/cropped',
        ground_truth_csv='ground_truth_train_split.csv'
    )
    svm_classifier.train_svm()
    
    # Brug den trænede SVM classifier i NeighbourDetector
    print("\nInitialiserer neighbour detector...")
    try:
        detector = NeighbourDetector(
            input_folder='splitted_dataset/train/cropped',
            classifier=svm_classifier
        )
        detector.process_images()
    except Exception as e:
        print(f"Fejl: {e}")