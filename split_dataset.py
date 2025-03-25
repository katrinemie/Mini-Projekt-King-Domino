import os
import shutil
import random

# Ny mappe til opdeling af billeder
base_dir = os.getcwd()  # Henter den nuværende arbejdsmappe
splitted_dataset_dir = os.path.join(base_dir, "splitted_dataset")

# Mapper for de opdelte billeder
train_full_dir = os.path.join(splitted_dataset_dir, "train", "full")
train_cropped_dir = os.path.join(splitted_dataset_dir, "train", "cropped")
test_full_dir = os.path.join(splitted_dataset_dir, "test", "full")
test_cropped_dir = os.path.join(splitted_dataset_dir, "test", "cropped")

# Mapperne oprettes, hvis de ikke allerede findes
for folder in [train_full_dir, train_cropped_dir, test_full_dir, test_cropped_dir]:
    os.makedirs(folder, exist_ok=True)

# Splitter billederne i træning og test
def split_images(source_dir, train_dir, test_dir, split_ratio=0.8):
    if not os.path.exists(source_dir):
        print(f"Fejl: Mappen {source_dir} findes ikke!")
        return
    
    images = os.listdir(source_dir)
    random.shuffle(images)  # Bland billederne tilfældigt

    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]

    # Kopier billeder til de rigtige mapper
    for img in train_images:
        try:
            shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))
        except Exception as e:
            print(f"Fejl ved kopi af {img} til træningsmappen: {e}")
    for img in test_images:
        try:
            shutil.copy(os.path.join(source_dir, img), os.path.join(test_dir, img))
        except Exception as e:
            print(f"Fejl ved kopi af {img} til testmappen: {e}")

# Split de billeder
original_full_dir = os.path.join(base_dir, "King Domino dataset", "Full game areas")
original_cropped_dir = os.path.join(base_dir, "King Domino dataset", "Cropped and perspective corrected boards")

split_images(original_full_dir, train_full_dir, test_full_dir)
split_images(original_cropped_dir, train_cropped_dir, test_cropped_dir)

print("Billederne er opdelt i træning og test, dab på den!")
