import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def ground_truth_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    ground_truth = {}

    for img_id in df['image_id'].unique():
        matrix = np.zeros((5, 5), dtype=int)
        subset = df[df['image_id'] == img_id]
        for _, row in subset.iterrows():
            col = int(row['x'] // 100)
            r = int(row['y'] // 100)
            matrix[r, col] = int(row['crowns'])
        ground_truth[f"{img_id}.jpg"] = matrix

    return ground_truth

class CrownDetector:
    def __init__(self, input_folder, template_paths, output_folder, scales, angles, threshold=0.6, highlight_color=(255, 182, 193)):
        self.input_folder = input_folder
        self.template_paths = template_paths
        self.output_folder = output_folder
        self.scales = scales
        self.angles = angles
        self.threshold = threshold
        self.highlight_color = highlight_color
        self.templates = self.load_templates()

        image_paths = glob.glob(os.path.join(self.input_folder, '*.jpg'))
        if not image_paths:
            raise FileNotFoundError("Ingen billeder fundet i input-mappen.")
        self.image_paths = image_paths

        os.makedirs(self.output_folder, exist_ok=True)

    def load_templates(self):
        templates = []
        for path in self.template_paths:
            template = cv2.imread(path)
            if template is None:
                print(f"⚠️ Kunne ikke finde template: {path}")
                continue
            templates.append(cv2.resize(template, (34, 29)))
        if not templates:
            raise ValueError("Ingen valide templates blev indlæst!")
        return templates

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        rot_mat[0, 2] += bound_w / 2 - center[0]
        rot_mat[1, 2] += bound_h / 2 - center[1]
        return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), borderValue=(255, 255, 255))

    def detect_crowns(self, board_img, filename):
        board_height, board_width = board_img.shape[:2]
        tile_height = board_height // 5
        tile_width = board_width // 5
        crown_counts = np.zeros((5, 5), dtype=int)

        for row in range(5):
            for col in range(5):
                x_start = col * tile_width
                y_start = row * tile_height
                tile_bgr = board_img[y_start:y_start + tile_height, x_start:x_start + tile_width]

                found_rects = []

                for template in self.templates:
                    for angle in self.angles:
                        rotated_template = self.rotate_image(template, angle)
                        for scale in self.scales:
                            resized_template = cv2.resize(rotated_template, (0, 0), fx=scale, fy=scale)
                            h, w = resized_template.shape[:2]

                            if h > tile_bgr.shape[0] or w > tile_bgr.shape[1]:
                                continue

                            result = cv2.matchTemplate(tile_bgr, resized_template, cv2.TM_CCOEFF_NORMED)
                            locations = np.where(result >= self.threshold)

                            for pt in zip(*locations[::-1]):
                                found_rects.append([pt[0], pt[1], w, h])

                rects = cv2.groupRectangles(found_rects, groupThreshold=1, eps=0.5)[0] if found_rects else []
                total_matches = len(rects)

                crown_counts[row, col] = total_matches
                if total_matches > 0:
                    print(f"{filename} - Tile ({row},{col}) - Fundne kroner efter NMS: {total_matches}")

        return crown_counts

    def process_images(self, ground_truth):
        y_true = []
        y_pred = []

        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            board_img = cv2.imread(img_path)
            if board_img is None:
                print(f"⚠️ Kunne ikke læse {filename}")
                continue

            crown_counts = self.detect_crowns(board_img, filename)

            if filename in ground_truth:
                gt_counts = ground_truth[filename]

                for r in range(5):
                    for c in range(5):
                        true_label = 1 if gt_counts[r, c] > 0 else 0
                        pred_label = 1 if crown_counts[r, c] > 0 else 0
                        y_true.append(true_label)
                        y_pred.append(pred_label)
            else:
                print(f"{filename} - Ingen ground truth tilgængelig")

        if y_true:
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
            acc = accuracy_score(y_true, y_pred) * 100

            print(f"\n✅ Samlet Crown Detection Accuracy: {acc:.2f}%")
            print("Confusion Matrix:\n", cm)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Crown', 'No Crown'], 
                        yticklabels=['Crown', 'No Crown'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix for Crown Detection')
            plt.tight_layout()
            plt.show()
        else:
            print("❌ Ingen data til beregning af confusion matrix.")

if __name__ == "__main__":
    detector = CrownDetector(
    input_folder='splitted_dataset/train/cropped',
    template_paths=[
        'board_templates/Skærmbillede 2025-04-23 kl. 13.08.35.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.08.47.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.08.56.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.09.05.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.09.13.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.09.25.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.09.44.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.09.51.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.10.19.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.11.34.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.12.43.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.13.03.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.13.14.png',
        'board_templates/Skærmbillede 2025-04-23 kl. 13.13.32.png'
    ],
    output_folder='output',
    scales=[0.8, 1.0, 1.2],
    angles=[0, 90, 180, 270],
    threshold=0.6
)


    # Indlæs ground truth fra CSV
    ground_truth = ground_truth_from_csv('ground_truth.csv')

    # Kør detektor
    detector.process_images(ground_truth)

