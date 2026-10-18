"""Model B: a lightweight CNN built from depthwise separable convolutions.

It must stay under 100,000 trainable parameters so it suits edge devices.

Run from the project root to print the layer summary and check the limit:
    python src/modelB.py
"""
import os

# These two settings must be made before TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # faster maths on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info and warning messages

import tensorflow as tf
from tensorflow.keras import layers, models

from data import augmentation, IMG_SIZE, NUM_CLASSES

PARAM_LIMIT = 100000  # maximum trainable parameters allowed by the assignment


def build_model_b():
    """Model B: depthwise separable convolutions (lightweight, under 100k parameters)."""
    model = models.Sequential(name="model_b_lightweight")
    model.add(layers.Input(shape=(*IMG_SIZE, 3)))  # 64 x 64 RGB image
    model.add(augmentation())                      # active during training only

    # Same four-block layout as Model A, but every convolution is depthwise separable:
    # a 3x3 filter on each channel separately, then a 1x1 convolution to mix channels.
    # This needs far fewer weights and multiplications than a normal 3x3 convolution.
    for filters in [32, 64, 128, 128]:
        # No bias: the BatchNormalization right after it already adds an offset
        model.add(layers.SeparableConv2D(filters, 3, padding="same", use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D(2))

    # Classifier head (same as Model A, with slightly less dropout)
    model.add(layers.GlobalAveragePooling2D())
    # Dropout randomly zeroes 20% of the pooled features each training step,
    # so the classifier cannot rely on any single feature (switched off at test time)
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64))
    model.add(layers.ReLU())
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))  # one probability per class
    return model


if __name__ == "__main__":
    model = build_model_b()
    model.summary()

    trainable = sum(int(tf.size(weight)) for weight in model.trainable_weights)
    status = "OK" if trainable <= PARAM_LIMIT else "TOO BIG"
    print(f"Trainable parameters: {trainable:,} (limit {PARAM_LIMIT:,}): {status}")
