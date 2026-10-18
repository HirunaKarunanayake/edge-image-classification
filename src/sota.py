"""Fine-tune a pretrained lightweight network (MobileNetV2 or EfficientNetB0) on the same split.

Training happens in two phases:
    1. The pretrained backbone is frozen and only the new classifier head trains.
    2. The whole network is unfrozen and trained further with a small learning rate.

Run from the project root:
    python src/sota.py <mobilenetv2|efficientnetb0>

Outputs:
    models/<name>_finetuned.weights.h5     best weights (lowest validation loss)
    results/<name>_finetuned_metrics.json  test metrics, size and timing
    plots/<name>_finetuned_curves.png      loss and accuracy curves
    plots/<name>_finetuned_confusion.png   confusion matrix
"""
import os

# These two settings must be made before TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # much faster maths on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info and warning messages

import sys
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from data import get_datasets, augmentation, IMG_SIZE, NUM_CLASSES, CLASSES
from evaluate import score_test_set
from plot_utils import plot_history
from train import EpochTimer

SEED = 42
HEAD_EPOCHS = 5   # phase 1: backbone frozen, train only the new classifier head
FINE_EPOCHS = 15  # phase 2: unfreeze everything, small learning rate

# Learning rates (Adam) and epochs for each pretrained model. Edit any value here.
#   head: phase 1, only the new classifier head trains (backbone frozen)
#   fine: phase 2, the whole network is unfrozen. Keep this rate small so the
#         pretrained features are adjusted, not overwritten.
RUN_CONFIG = {
    "mobilenetv2":    {"head_lr": 1e-3, "fine_lr": 1e-4, "head_epochs": HEAD_EPOCHS, "fine_epochs": FINE_EPOCHS},
    "efficientnetb0": {"head_lr": 1e-3, "fine_lr": 1e-4, "head_epochs": HEAD_EPOCHS, "fine_epochs": FINE_EPOCHS},
}


def build(name):
    """Build a pretrained backbone with a small new classifier head on top.

    Returns (full model, backbone); the backbone is returned too so it can be
    unfrozen later for phase 2.

    The data pipeline gives pixels in [0, 1], but each pretrained network expects
    its own input range, so a Rescaling layer inside the model converts it:
        MobileNetV2 expects [-1, 1], so pixels become 2 * x - 1
        EfficientNetB0 expects [0, 255], so pixels become 255 * x (it normalises internally)
    """
    shape = (*IMG_SIZE, 3)
    inputs = layers.Input(shape=shape)
    augmented = augmentation()(inputs)  # same training-only augmentation as Models A and B

    if name == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=shape)
        x = layers.Rescaling(2.0, offset=-1.0)(augmented)
    elif name == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=shape)
        x = layers.Rescaling(255.0)(augmented)
    else:
        raise ValueError("name must be 'mobilenetv2' or 'efficientnetb0'")

    backbone.trainable = False  # phase 1: frozen (batch-norm layers stay in inference mode)
    x = backbone(x)

    # New classifier head for our 6 classes (the original ImageNet head is not used)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name=name), backbone


def compile_model(model, lr):
    """Compile with Adam at learning rate lr (needed again after changing trainable)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",  # labels are class numbers 0 to 5
        metrics=["accuracy"],
    )


def make_checkpoint(path):
    """Save only the weights whenever the validation loss reaches a new lowest value."""
    return tf.keras.callbacks.ModelCheckpoint(
        path, monitor="val_loss", save_best_only=True, save_weights_only=True)


def main():
    # Read the command-line argument (default: MobileNetV2)
    name = sys.argv[1].lower() if len(sys.argv) > 1 else "mobilenetv2"
    tag = f"{name}_finetuned"
    tf.keras.utils.set_random_seed(SEED)

    train, val, test = get_datasets()  # same 70/15/15 split as Models A and B
    model, backbone = build(name)
    checkpoint_path = os.path.join("models", f"{tag}.weights.h5")

    config = RUN_CONFIG[name]
    print(f"{name}: phase 1 lr {config['head_lr']:g} x {config['head_epochs']} epochs, "
          f"phase 2 lr {config['fine_lr']:g} x {config['fine_epochs']} epochs")

    # Phase 1: train only the new head
    compile_model(model, config["head_lr"])
    head_timer = EpochTimer()
    head_history = model.fit(
        train, validation_data=val, epochs=config["head_epochs"],
        callbacks=[head_timer, make_checkpoint(checkpoint_path)], verbose=2)

    # Phase 2: unfreeze the whole network and fine-tune with a small learning rate
    backbone.trainable = True
    compile_model(model, config["fine_lr"])  # must recompile after changing trainable
    fine_timer = EpochTimer()
    fine_history = model.fit(
        train, validation_data=val, epochs=config["fine_epochs"],
        callbacks=[fine_timer, make_checkpoint(checkpoint_path)], verbose=2)

    model.load_weights(checkpoint_path)  # go back to the best validation-loss weights

    # Join the two phases into one history so the curves show every epoch
    history = {metric: head_history.history[metric] + fine_history.history[metric]
               for metric in head_history.history}
    plot_history(history, tag, frozen_epochs=config["head_epochs"])

    # Test-set evaluation
    confusion, precision, recall, accuracy = score_test_set(model, test, tag)

    # Resource numbers
    total_params = int(model.count_params())
    trainable_params = int(sum(int(tf.size(weight)) for weight in model.trainable_weights))
    size_mb = os.path.getsize(checkpoint_path) / 1024 ** 2  # weights-only file, no optimizer state
    estimate_mb = total_params * 4 / 1024 ** 2              # 4 bytes per float32 parameter
    epoch_time = float(np.mean(fine_timer.times))           # full fine-tuning epochs (phase 2)

    metrics = {
        "model": name,
        "run_config": config,
        "test_accuracy": accuracy,
        "precision_per_class": dict(zip(CLASSES, map(float, precision))),
        "recall_per_class": dict(zip(CLASSES, map(float, recall))),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion_matrix": confusion.tolist(),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb_weights_file": size_mb,
        "size_mb_float32_estimate": estimate_mb,
        "mean_epoch_time_s_phase2": epoch_time,
        "history": {metric: [float(value) for value in values]
                    for metric, values in history.items()},
    }
    with open(os.path.join("results", f"{tag}_metrics.json"), "w") as json_file:
        json.dump(metrics, json_file, indent=2)

    print()
    print(f"=== {name} (fine-tuned) ===")
    print(f"Test accuracy     : {accuracy:.4f}")
    print(f"Macro precision   : {precision.mean():.4f}")
    print(f"Macro recall      : {recall.mean():.4f}")
    print(f"Total params      : {total_params:,}  (trainable {trainable_params:,})")
    print(f"Model size        : {size_mb:.2f} MB  (float32 estimate {estimate_mb:.2f} MB)")
    print(f"Mean epoch time   : {epoch_time:.1f} s")
    print("Confusion matrix:")
    print(confusion)


if __name__ == "__main__":
    main()
