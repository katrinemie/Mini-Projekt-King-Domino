import cv2
import numpy as np
import os
import glob

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
                print(f" Kunne ikke finde template: {path}")
                continue
            templates.append(cv2.resize(template, (34, 29)))
        if not templates:
            raise ValueError(" Ingen valide templates blev indlæst!")
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
        return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), borderValue=(255,255,255))

    def process_images(self):
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            board_img = cv2.imread(img_path)
            if board_img is None:
                print(f" Kunne ikke læse {filename}")
                continue

            self.detect_crowns(board_img, filename)

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

                for (x, y, w, h) in rects:
                    top_left = (x_start + x, y_start + y)
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(board_img, top_left, bottom_right, self.highlight_color, 2)

                crown_counts[row, col] = total_matches
                if total_matches > 0:
                    print(f"{filename} - Tile ({row},{col}) - Fundne kroner efter NMS: {total_matches}")

        print(f"\n{filename} - Samlet kroner matrix:\n{crown_counts}\n")
        cv2.imwrite(os.path.join(self.output_folder, f"marked_{filename}"), board_img)
        cv2.imshow(f"Kroner: {filename}", board_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Main
if __name__ == "__main__":
    detector = CrownDetector(
        input_folder='splitted_dataset/train/cropped',
        template_paths=[
            'opdateret_skærmbillede.png',
            'opdateret_skærmbillede2.png',
            'opdateret_skærmbillede3.png',
            'opdateret_skærmbillede4.png'
        ],
        output_folder='outputs_with_crowns',
        scales=[0.9, 1.0, 1.2],
        angles=[0, 90, 180, 270],
        threshold=0.6
    )
    detector.process_images()
