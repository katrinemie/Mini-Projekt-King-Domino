import os
import shutil
import random

#path så de kommer til mapper
original_full_dir = "King Domino dataset/Full game areas"
original_cropped_dir = "King Domino dataset/Cropped and perspective corrected boards"

#path train/test mapper
train_full_dir = "King Domino dataset/train/full"
train_cropped_dir = "King Domino dataset/train/cropped"
test_full_dir = "King Domino dataset/test/full"
test_cropped_dir = "King Domino dataset/test/cropped"

# Opret mapperne hvis de ikke findes
for folder in [train_full_dir, train_cropped_dir, test_full_dir, test_cropped_dir]:
    os.makedirs(folder, exist_ok=True)

# Funktion til at opdele billeder
def split_images(source_dir, train_dir, test_dir, split_ratio=0.8):
    images = os.listdir(source_dir)
    random.shuffle(images)  # Bland billederne tilfældigt

    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]

    # Flyt billeder til de rigtige mapper
    for img in train_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(test_dir, img))

# Opdel billederne
split_images(original_full_dir, train_full_dir, test_full_dir)
split_images(original_cropped_dir, train_cropped_dir, test_cropped_dir)

print("Billederne er opdelt i træning og test, dab på den!")
