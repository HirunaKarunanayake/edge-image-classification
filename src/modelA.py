"""Model A: a standard CNN built from ordinary 2D convolutions.

Run from the project root to print the layer summary:
    python src/modelA.py
"""
import os

# These two settings must be made before TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # faster maths on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info and warning messages

from tensorflow.keras import layers, models

from data import augmentation, IMG_SIZE, NUM_CLASSES


def build_model_a():
    """Model A: standard 2D convolutions, each block followed by max-pooling."""
    model = models.Sequential(name="model_a_standard")
    model.add(layers.Input(shape=(*IMG_SIZE, 3)))  # 64 x 64 RGB image
    model.add(augmentation())                      # active during training only

    # Four blocks of Conv, BatchNorm, ReLU, MaxPool.
    # Each MaxPool halves the image size: 64, 32, 16, 8, 4 pixels.
    for filters in [32, 64, 128, 128]:
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=True))
        # momentum=0.9 (default 0.99) lets the running mean and variance used at
        # validation time keep up with the weights, which reduces val-loss spikes
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D(2))

    # Classifier head.
    # Global average pooling turns each feature map into one number, which is much
    # smaller than flattening everything into a large dense layer.
    model.add(layers.GlobalAveragePooling2D())
    # Dropout randomly zeroes 30% of the pooled features each training step,
    # so the classifier cannot rely on any single feature (switched off at test time)
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64))
    model.add(layers.ReLU())
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))  # one probability per class
    return model


if __name__ == "__main__":
    build_model_a().summary()
