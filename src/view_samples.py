"""Save a grid of 4 random training images for every class to plots/samples.png.

Run from the project root:
    python src/view_samples.py
"""
import os
import random

import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = os.path.join("data", "seg_train", "seg_train")
IMAGES_PER_CLASS = 4

class_names = sorted(os.listdir(TRAIN_DIR))

# One row per class, one column per sample image
fig, axes = plt.subplots(len(class_names), IMAGES_PER_CLASS, figsize=(8, 12))

for row, class_name in enumerate(class_names):
    class_dir = os.path.join(TRAIN_DIR, class_name)
    sample_files = random.sample(os.listdir(class_dir), IMAGES_PER_CLASS)

    for col, file_name in enumerate(sample_files):
        ax = axes[row, col]
        ax.imshow(Image.open(os.path.join(class_dir, file_name)))
        ax.axis("off")
        if col == 0:
            ax.set_title(class_name, fontsize=9, loc="left")  # label the row once

plt.tight_layout()
plt.savefig(os.path.join("plots", "samples.png"), dpi=120)
plt.show()
