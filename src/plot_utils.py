"""Plotting helpers shared by train.py, sota.py and evaluate.py.

Figures are saved into the plots/ folder instead of being shown on screen.
"""
import os

import matplotlib
matplotlib.use("Agg")  # draw straight to image files; no pop-up window
import matplotlib.pyplot as plt

from data import CLASSES, NUM_CLASSES


def plot_history(history, tag, frozen_epochs=None):
    """Plot training and validation loss and accuracy for every epoch.

    history:       dict of per-epoch lists, such as the .history of a Keras fit() result
    tag:           run name used in the titles and the file name
    frozen_epochs: for fine-tuned models (sota.py), the number of epochs trained with the
                   pretrained backbone frozen; a dashed line marks where it was unfrozen

    Saves plots/<tag>_curves.png and returns that path.
    """
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: loss. Right panel: accuracy.
    panels = [(axes[0], "loss", "Cross-entropy loss"),
              (axes[1], "accuracy", "Accuracy")]
    for ax, metric, y_label in panels:
        # "b-o" is a blue line with dots, "r-o" is a red line with dots
        ax.plot(epochs, history[metric], "b-o", markersize=3, label=f"Training {metric}")
        ax.plot(epochs, history["val_" + metric], "r-o", markersize=3, label=f"Validation {metric}")
        if frozen_epochs is not None:
            # Halfway between the last frozen epoch and the first fine-tuning epoch
            ax.axvline(frozen_epochs + 0.5, color="gray", linestyle="--", label="Backbone unfrozen")
        ax.set(title=f"{tag}: {metric}", xlabel="Epoch", ylabel=y_label)
        ax.legend()

    plt.tight_layout()
    path = os.path.join("plots", f"{tag}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(confusion, tag):
    """Draw a confusion matrix with the image count written in every cell.

    Rows are the true classes and columns are the predicted classes.
    Saves plots/<tag>_confusion.png.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")  # darker blue means more images

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{tag}: confusion matrix (test)")

    # White text on dark cells and black text on light cells, so every number is readable
    dark_threshold = confusion.max() / 2
    for true_class in range(NUM_CLASSES):
        for predicted_class in range(NUM_CLASSES):
            count = int(confusion[true_class, predicted_class])
            text_color = "white" if count > dark_threshold else "black"
            ax.text(predicted_class, true_class, count,
                    ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{tag}_confusion.png"), dpi=150)
    plt.close(fig)
