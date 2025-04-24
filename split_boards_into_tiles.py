import os
import cv2

def split_board_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue

        board_id = os.path.splitext(filename)[0]  # fx '1'
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Kunne ikke åbne {filename}")
            continue

        height, width, _ = image.shape
        tile_h = height // 5
        tile_w = width // 5

        for i in range(5):
            for j in range(5):
                x = j * tile_w
                y = i * tile_h
                tile = image[y:y+tile_h, x:x+tile_w]

                # Brug image ID + x/y som filnavn
                tile_filename = f"tile_{board_id}_{x}_{y}.jpg"
                tile_path = os.path.join(output_folder, tile_filename)
                cv2.imwrite(tile_path, tile)

        print(f"✅ Split: {filename} → 25 tiles gemt som tile_{board_id}_x_y.jpg")

# === Brug det her ===
input_folder = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\Mini projekt king domino\Mini-Projekt-King-Domino\King Domino dataset\Cropped and perspective corrected boards"
output_folder = r"C:\Users\katri\Desktop\Kingkat\All_Tiles"

split_board_images(input_folder, output_folder)
