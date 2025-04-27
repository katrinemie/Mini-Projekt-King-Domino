import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



class NeighbourDetection:
    # Initialiserer objektet og finder billeder i inputmappen
    # Sætter terræntyper og nabo-retninger
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        
        if not self.image_paths:
            print("Ingen billeder fundet i inputmappen.")
        
        #terræntype farver og klasser
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
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  #Op, ned, venstre, højre

    # Klassificerer en tile baseret på farve
    def classify_tile(self, tile):
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

    # Opdeler billedet i et 5x5 grid
    def split_to_tiles(self, image):
        h, w = image.shape[:2]
        return [np.hsplit(row, 5) for row in np.vsplit(image, 5)]

    # Finder naboer af samme type og tæller dem
    def find_neighbours(self, labels):
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

    # Visualiserer originalbilledet og naboanalyse
    def visualize_results(self, image, labels, connections, neighbour_counts):
        plt.figure(figsize=(14, 6))
        
        # Originalbillede
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Originalbillede')
        plt.axis('off')
        
        #Visualisering af naboer
        plt.subplot(1, 2, 2)
        h, w = image.shape[:2]
        tile_h, tile_w = h // 5, w // 5
        
        
        for r in range(5):
            for c in range(5):
                color = np.array(self.terrain_colors[labels[r][c]])/255
                rect = plt.Rectangle((c*tile_w, r*tile_h), tile_w, tile_h, 
                                    facecolor=color, edgecolor='white', alpha=0.7)
                plt.gca().add_patch(rect)
                
                #antal naboer
                plt.text(c*tile_w + tile_w/2, r*tile_h + tile_h/2,
                        str(neighbour_counts[r, c]),
                        ha='center', va='center', fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        #Tegner linjer mellem forbundne naboer
        for (r1, c1), (r2, c2) in connections:
            plt.plot([c1*tile_w + tile_w/2, c2*tile_w + tile_w/2],
                    [r1*tile_h + tile_h/2, r2*tile_h + tile_h/2],
                    'white', linewidth=1.5, alpha=0.7)
        
        #Opret legend i den ønskede rækkefølge 
        desired_order = ["Field", "Forest", "Lake", "Grassland", "Swamp", "Mine", "Home", "Unknown"]
        legend_elements = [Patch(facecolor=np.array(self.terrain_colors[name])/255, label=name) 
                         for name in desired_order]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Naboanalyse (Antal naboer af samme type)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    # Behandler alle billeder i inputmappen og visualiserer naboanalyserne
    def process_images(self):
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            #Klassificer tiles
            tiles = self.split_to_tiles(img)
            labels = [[self.classify_tile(tile) for tile in row] for row in tiles]
            
            #Finder naboer
            connections, neighbour_counts = self.find_neighbours(labels)
            
            print(f"\nAnalyserer {os.path.basename(path)}:")
            self.visualize_results(img, labels, connections, neighbour_counts)


if __name__ == "__main__":
    INPUT_FOLDER = 'splitted_dataset/train/cropped'  
    try:
        detector = NeighbourDetection(INPUT_FOLDER)
        detector.process_images()
    except Exception as e:
        print(f"Fejl: {e}")

    