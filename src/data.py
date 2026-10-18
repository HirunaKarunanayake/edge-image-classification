"""Data pipeline: reads the fixed split from results/split.csv and builds tf.data datasets.

Every image is decoded as RGB, resized to IMG_SIZE and scaled to the range [0, 1].

Run from the project root to print one batch from each subset:
    python src/data.py
"""
import os

# These two settings must be made before TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # faster maths on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info and warning messages

import csv
from pathlib import PureWindowsPath

import tensorflow as tf

IMG_SIZE = (64, 64)  # every image is resized to 64x64 pixels
BATCH_SIZE = 32
SEED = 42

# Class names in label order (label 0 is "buildings", label 5 is "street").
# This matches the alphabetical folder order used by split_data.py.
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASSES)

AUTOTUNE = tf.data.AUTOTUNE  # let TensorFlow choose how much work to do in parallel
SPLIT_CSV = os.path.join("results", "split.csv")


def read_split(csv_path=SPLIT_CSV):
    """Read split.csv and group the image paths and labels by subset.

    Returns {"train": (paths, labels), "val": (paths, labels), "test": (paths, labels)}.
    """
    data = {subset: ([], []) for subset in ["train", "val", "test"]}
    with open(csv_path) as csv_file:
        for row in csv.DictReader(csv_file):
            # split.csv was written on Windows, so its paths use Windows separators.
            # Split the path into folder names and rejoin them for this operating system.
            folders = PureWindowsPath(row["path"]).parts
            path = os.path.join(*folders)

            paths, labels = data[row["subset"]]
            paths.append(path)
            labels.append(int(row["label"]))
    return data


def load_image(path, label):
    """Load one JPEG file as a float image with values in [0, 1]."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # always 3 colour channels (RGB)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0       # pixel values 0..255 become 0..1
    return image, label


def make_dataset(paths, labels, training):
    """Build a batched dataset. Only the training data is shuffled."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Load images in parallel, and keep them in memory after the first epoch
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE).cache()
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED)  # new order every epoch
    # prefetch prepares the next batch while the model is busy with the current one
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def augmentation():
    """Random flips, rotations and zooms, applied only while training.

    Keras switches these layers off at validation, test and inference time, so the
    model sees a slightly different version of every training image each epoch
    (less memorising) while evaluation stays the same every time. They have no
    weights, so they do not change the parameter count.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),  # a mirrored scene is still the same scene
        tf.keras.layers.RandomRotation(0.05, seed=SEED),      # rotate up to 18 degrees either way
        tf.keras.layers.RandomZoom(0.1, seed=SEED),           # zoom in or out by up to 10 percent
    ], name="augmentation")


def get_datasets():
    """Return the (train, val, test) datasets used by every training script."""
    split = read_split()
    train = make_dataset(*split["train"], training=True)
    val = make_dataset(*split["val"], training=False)
    test = make_dataset(*split["test"], training=False)
    return train, val, test


if __name__ == "__main__":
    # Quick check: shape, value range and first 8 labels of one batch from each subset
    train, val, test = get_datasets()
    for name, dataset in [("train", train), ("val", val), ("test", test)]:
        images, labels = next(iter(dataset))
        smallest = float(tf.reduce_min(images))
        largest = float(tf.reduce_max(images))
        print(name, images.shape, images.dtype, smallest, largest, labels[:8].numpy())
