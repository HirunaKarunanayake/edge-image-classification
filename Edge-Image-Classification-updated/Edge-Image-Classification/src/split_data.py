"""Make one fixed 70 / 15 / 15 train / validation / test split and save it to results/split.csv.

The Kaggle train and test folders are merged first, then every class is split
on its own, so all three subsets keep the same class balance. Every model
reads this file, so they are all trained and tested on exactly the same images.

Run from the project root:
    python src/split_data.py
"""
import csv
import os
import random
from collections import Counter

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # whatever is left over (about 15%) becomes the test set
SUBSETS = ["train", "val", "test"]

SOURCE_DIRS = [
    os.path.join("data", "seg_train", "seg_train"),
    os.path.join("data", "seg_test", "seg_test"),
]
OUTPUT_CSV = os.path.join("results", "split.csv")

random.seed(SEED)  # same shuffle every time the script runs

# Class folders are sorted alphabetically; the position in this list is the label number
class_names = sorted(os.listdir(SOURCE_DIRS[0]))

rows = []                # one (path, label, class name, subset) row per image
subset_counts = Counter()  # images per (class name, subset), for the summary below

for label, class_name in enumerate(class_names):
    # Collect this class's images from both folders.
    # Sorting first makes the shuffle reproducible on any computer.
    files = []
    for source_dir in SOURCE_DIRS:
        class_dir = os.path.join(source_dir, class_name)
        files += [os.path.join(class_dir, name) for name in sorted(os.listdir(class_dir))]
    random.shuffle(files)

    n_train = int(TRAIN_FRACTION * len(files))
    n_val = int(VAL_FRACTION * len(files))

    # After shuffling: the first n_train files go to train, the next n_val to val, the rest to test
    for index, path in enumerate(files):
        if index < n_train:
            subset = "train"
        elif index < n_train + n_val:
            subset = "val"
        else:
            subset = "test"
        rows.append((path, label, class_name, subset))
        subset_counts[(class_name, subset)] += 1

with open(OUTPUT_CSV, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["path", "label", "class", "subset"])
    writer.writerows(rows)

# Summary: images per class in each subset, then the totals
for class_name in class_names:
    train_count = subset_counts[(class_name, "train")]
    val_count = subset_counts[(class_name, "val")]
    test_count = subset_counts[(class_name, "test")]
    print(f"{class_name:10s} train {train_count}  val {val_count}  test {test_count}")

totals = {subset: sum(subset_counts[(name, subset)] for name in class_names) for subset in SUBSETS}
print("total", totals)
