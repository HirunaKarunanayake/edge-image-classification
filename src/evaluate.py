"""Evaluate trained models on the test set and build the comparison tables.

Run from the project root:
    python src/evaluate.py [tag ...]
    python src/evaluate.py A_adam B_adam
With no tags, every model saved in models/*.keras is evaluated.

Outputs:
    results/<tag>_test_metrics.json   accuracy, precision, recall, size, speed, ...
    plots/<tag>_confusion.png         confusion matrix
    results/comparison.md             Markdown tables for the report
"""
import os

# These two settings must be made before TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # faster maths on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info and warning messages

import sys
import json
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data import get_datasets, CLASSES, NUM_CLASSES
from plot_utils import plot_confusion


def count_macs(model):
    """Count the multiply-accumulate operations (MACs) needed to classify one image.

    Only convolution, separable convolution and dense layers are counted;
    pooling, batch normalisation and activations are cheap in comparison.
    """
    macs = 0
    for layer in model.layers:
        if isinstance(layer, layers.SeparableConv2D):
            height, width, out_channels = layer.output.shape[1:]  # skip the batch dimension
            kernel_area = layer.kernel_size[0] * layer.kernel_size[1]
            in_channels = layer.input.shape[-1]
            macs += height * width * in_channels * kernel_area   # depthwise: one 3x3 filter per channel
            macs += height * width * in_channels * out_channels  # pointwise: 1x1 filters mix the channels
        elif isinstance(layer, layers.Conv2D):
            height, width, out_channels = layer.output.shape[1:]
            kernel_area = layer.kernel_size[0] * layer.kernel_size[1]
            in_channels = layer.input.shape[-1]
            macs += height * width * in_channels * kernel_area * out_channels
        elif isinstance(layer, layers.Dense):
            macs += layer.input.shape[-1] * layer.units
    return int(macs)


def score_test_set(model, test, tag):
    """Predict every test image, save the confusion matrix plot, and return the scores.

    Returns (confusion, precision, recall, accuracy); precision and recall have one
    value per class.
    """
    true_labels = np.concatenate([labels.numpy() for images, labels in test])
    predicted_labels = model.predict(test, verbose=0).argmax(axis=1)  # most likely class

    # confusion[i, j] = number of class-i images that were predicted as class j
    confusion = tf.math.confusion_matrix(
        true_labels, predicted_labels, num_classes=NUM_CLASSES).numpy()
    correct = np.diag(confusion)  # correctly classified images per class

    # Precision: of the images predicted as a class, the fraction that really were that class.
    # Recall:    of the images that really are a class, the fraction the model found.
    # np.maximum(..., 1) avoids dividing by zero for a class that never appears.
    precision = correct / np.maximum(confusion.sum(axis=0), 1)
    recall = correct / np.maximum(confusion.sum(axis=1), 1)
    accuracy = float(correct.sum() / confusion.sum())

    plot_confusion(confusion, tag)
    return confusion, precision, recall, accuracy


def evaluate(tag, test):
    """Compute and save the test-set metrics for one model trained by train.py.

    tag is the run name, e.g. "A_adam". The model is loaded from models/<tag>.keras
    and its training history from results/<tag>_history.json.
    """
    model_path = os.path.join("models", f"{tag}.keras")
    model = tf.keras.models.load_model(model_path)
    confusion, precision, recall, accuracy = score_test_set(model, test, tag)

    # Training statistics saved by train.py
    with open(os.path.join("results", f"{tag}_history.json")) as json_file:
        training = json.load(json_file)

    # Epoch 1 is slower because it also fills the image cache, so leave it out of the average
    epoch_times = training["epoch_times"]
    if len(epoch_times) > 1:
        mean_epoch_time = float(np.mean(epoch_times[1:]))
    else:
        mean_epoch_time = float(epoch_times[0])

    # The saved checkpoint is the epoch with the lowest validation loss
    val_loss = training["history"]["val_loss"]
    best = int(np.argmin(val_loss))  # counted from 0

    total_params = int(model.count_params())
    trainable_params = int(sum(int(tf.size(weight)) for weight in model.trainable_weights))

    metrics = {
        "model": tag,
        "test_accuracy": accuracy,
        "precision_per_class": dict(zip(CLASSES, map(float, precision))),
        "recall_per_class": dict(zip(CLASSES, map(float, recall))),
        "macro_precision": float(precision.mean()),  # plain average over the 6 classes
        "macro_recall": float(recall.mean()),
        "confusion_matrix": confusion.tolist(),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs": count_macs(model),
        # The .keras file also stores the optimizer state; the float32 estimate
        # (4 bytes per parameter) is the size of the weights alone
        "size_kb_keras_file": os.path.getsize(model_path) / 1024,
        "size_kb_float32_estimate": total_params * 4 / 1024,
        "mean_epoch_time_s": mean_epoch_time,
        "best_epoch": best + 1,  # counted from 1, as in the training log
        "best_val_loss": float(val_loss[best]),
        "best_val_accuracy": float(training["history"]["val_accuracy"][best]),
    }
    with open(os.path.join("results", f"{tag}_test_metrics.json"), "w") as json_file:
        json.dump(metrics, json_file, indent=2)

    print(f"{tag:16s} test acc {accuracy:.4f}  "
          f"macro P {precision.mean():.4f}  macro R {recall.mean():.4f}")
    return metrics


def load_metrics(pattern):
    """Load every results/<pattern> JSON file into a dict keyed by model name."""
    metrics_by_model = {}
    for path in sorted(glob.glob(os.path.join("results", pattern))):
        with open(path) as json_file:
            metrics = json.load(json_file)
        metrics_by_model[metrics["model"]] = metrics
    return metrics_by_model


def markdown_table(header, rows):
    """Build a Markdown table and return it as a list of text lines.

    header is a list of column titles; rows is a list of rows, each a list of cell texts.
    """
    def table_line(cells):
        return "| " + " | ".join(cells) + " |"

    lines = [table_line(header), table_line(["---"] * len(header))]
    lines += [table_line(row) for row in rows]
    return lines


def write_tables():
    """Collect all saved metrics into Markdown tables and save results/comparison.md."""
    own_models = load_metrics("*_test_metrics.json")      # Models A and B, from evaluate()
    pretrained = load_metrics("*_finetuned_metrics.json")  # fine-tuned models, from sota.py
    lines = []

    # Table 1: Model A against Model B, both trained with Adam
    rows = []
    for tag in ["A_adam", "B_adam"]:
        if tag in own_models:
            metrics = own_models[tag]
            rows.append([
                tag[0],  # "A" or "B"
                f"{metrics['total_params']:,}",
                f"{metrics['trainable_params']:,}",
                f"{metrics['macs']:,}",
                f"{metrics['size_kb_float32_estimate']:.1f}",
                f"{metrics['size_kb_keras_file']:.1f}",
                f"{metrics['mean_epoch_time_s']:.1f}",
                f"{metrics['test_accuracy']:.4f}",
                f"{metrics['macro_precision']:.4f}",
                f"{metrics['macro_recall']:.4f}",
            ])
    lines += ["## Model A vs Model B (Adam)", ""]
    lines += markdown_table(
        ["Model", "Total params", "Trainable params", "MACs / image", "Size (KB, float32)",
         "Size (KB, .keras file)", "Epoch time (s)", "Test acc", "Macro precision", "Macro recall"],
        rows)

    # Table 2: every optimizer run
    rows = []
    for tag, metrics in own_models.items():
        rows.append([
            tag,
            str(metrics["best_epoch"]),
            f"{metrics['best_val_loss']:.4f}",
            f"{metrics['best_val_accuracy']:.4f}",
            f"{metrics['test_accuracy']:.4f}",
        ])
    lines += ["", "## Optimizer comparison", ""]
    lines += markdown_table(
        ["Run", "Best epoch", "Best val loss", "Best val acc", "Test acc"], rows)

    # Table 3: Model B against the fine-tuned pretrained models (only if sota.py has been run)
    if pretrained:
        rows = []
        if "B_adam" in own_models:
            metrics = own_models["B_adam"]
            rows.append([
                "Model B",
                f"{metrics['total_params']:,}",
                f"{metrics['size_kb_float32_estimate'] / 1024:.3f}",  # KB to MB
                f"{metrics['mean_epoch_time_s']:.1f}",
                f"{metrics['test_accuracy']:.4f}",
                f"{metrics['macro_precision']:.4f}",
                f"{metrics['macro_recall']:.4f}",
            ])
        for name, metrics in pretrained.items():
            rows.append([
                name,
                f"{metrics['total_params']:,}",
                f"{metrics['size_mb_float32_estimate']:.2f}",
                f"{metrics['mean_epoch_time_s_phase2']:.1f}",
                f"{metrics['test_accuracy']:.4f}",
                f"{metrics['macro_precision']:.4f}",
                f"{metrics['macro_recall']:.4f}",
            ])
        lines += ["", "## Model B vs fine-tuned lightweight SOTA", ""]
        lines += markdown_table(
            ["Model", "Total params", "Size (MB)", "Epoch time (s)",
             "Test acc", "Macro precision", "Macro recall"],
            rows)

    path = os.path.join("results", "comparison.md")
    with open(path, "w") as markdown_file:
        markdown_file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print(f"Saved {path}")


def main():
    # Model tags from the command line, or every saved model if none are given
    tags = sys.argv[1:]
    if not tags:
        model_files = sorted(glob.glob(os.path.join("models", "*.keras")))
        # "models/A_adam.keras" becomes "A_adam"
        tags = [os.path.splitext(os.path.basename(path))[0] for path in model_files]

    train, val, test = get_datasets()  # only the test set is used here
    for tag in tags:
        evaluate(tag, test)
    write_tables()


if __name__ == "__main__":
    main()
