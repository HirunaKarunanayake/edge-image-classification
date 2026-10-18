"""Count how many images each class has in the original Kaggle train and test folders.

Run from the project root:
    python src/check_data.py
"""
import os

DATA_ROOT = "data"

# The Kaggle archive puts each split inside a folder with the same name,
# for example data/seg_train/seg_train/forest/123.jpg
for split in ["seg_train", "seg_test"]:
    split_dir = os.path.join(DATA_ROOT, split, split)
    print(split)

    total = 0
    for class_name in sorted(os.listdir(split_dir)):
        image_count = len(os.listdir(os.path.join(split_dir, class_name)))
        total += image_count
        print(f"  {class_name:10s} {image_count}")

    print(f"  {'total':10s} {total}")
    print()  # blank line between the two splits
