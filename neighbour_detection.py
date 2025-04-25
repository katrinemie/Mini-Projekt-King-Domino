import cv2
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class EnhancedTileAnalyzer(TileAnalyzer):
    def __init__(self, input_folder, ground_truth_folder):
        super().__init__(input_folder, ground_truth_folder)
        # Definer farver for hver terræntype
        self.terrain_colors = {
            "Field": (255, 215, 0),       # Guld
            "Forest": (34, 139, 34),      # Skovgrøn
            "Lake": (65, 105, 225),       # Kongeblå
            "Grassland": (152, 251, 152), # Lysegrøn
            "Swamp": (139, 137, 112),    # Sumpskimmel
            "Mine": (169, 169, 169),      # Mørkegrå
            "Home": (255, 99, 71),        # Tomat
            "Unknown": (220, 220, 220)   # Lysgrå
        }
        
    def create_terrain_map(self, labels):
        """Opret et farvekort over terræntyper"""
        terrain_map = np.zeros((5, 5, 3), dtype=np.uint8)
        for r in range(5):
            for c in range(5):
                terrain_map[r, c] = self.terrain_colors.get(labels[r][c], (220, 220, 220))
        return terrain_map
    
    def visualize_regions(self, image, regions, labels):
        """Forbedret visualisering med matplotlib"""
        plt.figure(figsize=(12, 8))
        
        # Terrænkort
        plt.subplot(1, 2, 1)
        terrain_map = self.create_terrain_map(labels)
        plt.imshow(terrain_map)
        plt.title('Terræntype Kort')
        plt.axis('off')
        
        # Tilføj farveforklaring
        unique_labels = set(labels[r][c] for r in range(5) for c in range(5))
        legend_elements = [plt.Rectangle((0,0),1,1, color=np.array(self.terrain_colors[l])/255, 
                          label=l) for l in unique_labels]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Regioner med scores
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Regionsvisualisering')
        plt.axis('off')
        
        h, w = image.shape[:2]
        tile_h, tile_w = h // 5, w // 5
        
        for region in regions:
            # Tegn regionkonturer
            mask = np.zeros((5, 5), dtype=np.uint8)
            for r, c in region["tiles"]:
                mask[r, c] = 1
            
            # Find konturer
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt = cnt.squeeze() * [tile_w, tile_h] + [tile_w//2, tile_h//2]
                plt.plot(cnt[:, 0], cnt[:, 1], 'w-', linewidth=2)
            
            # Tilføj tekstlabel
            center_r = sum(r for r, _ in region["tiles"]) / len(region["tiles"])
            center_c = sum(c for _, c in region["tiles"]) / len(region["tiles"])
            plt.text(center_c * tile_w, center_r * tile_h, 
                    f"{region['type']}\nTiles: {len(region['tiles'])}\nCrowns: {region['crowns']}\nScore: {region['score']}",
                    color='white', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        
        plt.tight_layout()
        plt.show()
    
    def process_images(self):
        all_true = []
        all_pred = []

        for path in self.image_paths:
            print(f"\n📄 Behandler billede: {os.path.basename(path)}")
            img = cv2.imread(path)
            if img is None:
                print("⚠️ Kunne ikke læse billedet.")
                continue

            labels, crowns = self.process_image(img)
            regions = self.find_regions_and_score(labels, crowns)

            # Hent ground truth-billede
            gt_path = os.path.join(self.ground_truth_folder, os.path.basename(path))
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                if gt_img is not None:
                    gt_labels, _ = self.process_image(gt_img)
                    for r in range(5):
                        for c in range(5):
                            all_true.append(gt_labels[r][c])
                            all_pred.append(labels[r][c])

            # Visuelt output
            self.visualize_regions(img, regions, labels)

            # Tekst-output
            print("\n🧩 Fundne regioner:")
            for i, region in enumerate(regions):
                print(f"Region {i+1}: {region['type']} – {len(region['tiles'])} tiles – {region['crowns']} kroner – Score: {region['score']}")

        # Evaluer nøjagtighed
        if all_true and all_pred:
            labels_order = ["Field", "Forest", "Lake", "Grassland", "Swamp", "Mine", "Home", "Unknown"]
            cm = confusion_matrix(all_true, all_pred, labels=labels_order)
            acc = accuracy_score(all_true, all_pred)

            print("\n📊 Confusion Matrix:")
            print(cm)
            print(f"\n✅ Accuracy: {acc * 100:.2f}%")

            # Visualiser confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xticks(range(len(labels_order)), labels_order, rotation=45)
            plt.yticks(range(len(labels_order)), labels_order)
            plt.colorbar()
            
            # Tilføj tekstværdier
            for i in range(len(labels_order)):
                for j in range(len(labels_order)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            
            plt.tight_layout()
            plt.show()

# === Main Execution ===
if __name__ == "__main__":
    print("▶️ Starter forbedret analyse med regionsscoring og visualisering...\n")

    INPUT_FOLDER = 'splitted_dataset/train/cropped'  # Tilpas denne
    GROUND_TRUTH_FOLDER = 'ground_truth_images'       # Tilpas denne

    analyzer = EnhancedTileAnalyzer(INPUT_FOLDER, GROUND_TRUTH_FOLDER)
    analyzer.process_images()