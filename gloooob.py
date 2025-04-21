import cv2
import numpy as np
import os
import glob

# === Glob-kontrol ===
input_folder = 'splitted_dataset/train/cropped'
template_folder = 'crown_templates'

# Find alle .jpg-billeder i din input-mappe
image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))

# Find alle .png-templates i din template-mappe
template_paths = glob.glob(os.path.join(template_folder, '*.png'))

print("🔍 Fundne billeder (.jpg):")
if image_paths:
    for path in image_paths:
        print(" -", path)
else:
    print("Ingen billeder fundet i:", input_folder)

print("\nFundne templates (.png):")
if template_paths:
    for path in template_paths:
        print(" -", path)
else:
    print("Ingen templates fundet i:", template_folder)
