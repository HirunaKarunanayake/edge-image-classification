"""List the image sizes and colour modes found in the training folder.

Most images in this dataset are 150x150 RGB, but not all of them, which is why
data.py resizes every image and forces 3 colour channels.

Run from the project root:
    python src/check_sizes.py
"""
import os
from collections import Counter

from PIL import Image

TRAIN_DIR = os.path.join("data", "seg_train", "seg_train")

print("Scanning training images...")

# Number of images for each ((width, height), colour mode) combination
size_counts = Counter()

for class_name in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, class_name)
    for file_name in os.listdir(class_dir):
        # PIL only reads the file header here, so this is quick
        with Image.open(os.path.join(class_dir, file_name)) as img:
            size_counts[(img.size, img.mode)] += 1

# Most common combination first
for (size, mode), count in size_counts.most_common():
    print(size, mode, count)
