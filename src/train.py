"""Train Model A or Model B with one optimizer and save the best checkpoint.

Run from the project root:
    python src/train.py <A|B> <adam|sgd|sgd_momentum> [epochs]
    python src/train.py A adam
    python src/train.py B sgd_momentum 40

Outputs:
    models/<model>_<optimizer>.keras          best weights (lowest validation loss)
    results/<model>_<optimizer>_history.json  per-epoch numbers for the report
    plots/<model>_<optimizer>_curves.png      loss and accuracy curves
"""
import os

# These two settings must be made before TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # faster maths on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info and warning messages

import sys
import json
import time

import tensorflow as tf

from data import get_datasets
from modelA import build_model_a
from modelB import build_model_b
from plot_utils import plot_history

SEED = 42
MIN_EPOCHS = 20  # assignment minimum; EarlyStopping never stops before this

# Settings for every (model, optimizer) run. Edit any value here.
#   lr:     starting learning rate. ReduceLROnPlateau lowers it during training;
#           the rate used in each epoch is printed and saved in
#           results/<model>_<optimizer>_history.json.
#   epochs: number of epochs (keep it at least MIN_EPOCHS for the assignment).
# A third command-line argument overrides epochs for one run, for example:
#   python src/train.py A adam 40
RUN_CONFIG = {
    ("A", "adam"):         {"lr": 1e-3, "epochs": 20},
    ("A", "sgd"):          {"lr": 5e-3, "epochs": 20},
    ("A", "sgd_momentum"): {"lr": 5e-3, "epochs": 20},
    ("B", "adam"):         {"lr": 1e-3, "epochs": 20},
    ("B", "sgd"):          {"lr": 1e-2, "epochs": 20},
    ("B", "sgd_momentum"): {"lr": 1e-2, "epochs": 20},
}
MOMENTUM = 0.9  # used by sgd_momentum only


def get_optimizer(name, lr):
    """Create the optimizer called name ("adam", "sgd" or "sgd_momentum")."""
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "sgd_momentum":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=MOMENTUM)
    raise ValueError(f"Unknown optimizer: {name}")


class LRLogger(tf.keras.callbacks.Callback):
    """Records and prints the learning rate used in every epoch."""

    def on_train_begin(self, logs=None):
        self.lrs = []

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate.numpy())
        self.lrs.append(lr)
        print(f"Epoch {epoch + 1}: learning rate = {lr:.6g}")


class EpochTimer(tf.keras.callbacks.Callback):
    """Records how many seconds every epoch takes (needed for the comparison table)."""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_start)


def main():
    # Read the command-line arguments (defaults: model A with Adam)
    model_name = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    opt_name = sys.argv[2].lower() if len(sys.argv) > 2 else "adam"
    tag = f"{model_name}_{opt_name}"  # run name, e.g. "A_adam"

    if (model_name, opt_name) not in RUN_CONFIG:
        raise ValueError(f"No settings in RUN_CONFIG for model {model_name} with {opt_name}")
    config = RUN_CONFIG[(model_name, opt_name)]
    lr = config["lr"]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else config["epochs"]

    if epochs < MIN_EPOCHS:
        print(f"Note: {epochs} epochs is below the assignment minimum of {MIN_EPOCHS}")
    print(f"Training model {model_name} with {opt_name}: starting lr {lr:g}, {epochs} epochs")

    tf.keras.utils.set_random_seed(SEED)  # same starting weights and shuffle order every run

    train, val, test = get_datasets()  # the test set is only used by evaluate.py
    model = build_model_a() if model_name == "A" else build_model_b()

    model.compile(
        optimizer=get_optimizer(opt_name, lr),
        loss="sparse_categorical_crossentropy",  # labels are class numbers 0 to 5
        metrics=["accuracy"],
    )

    # Callbacks run at the start and end of every epoch
    timer = EpochTimer()
    lr_log = LRLogger()

    # Save the model whenever the validation loss reaches a new lowest value
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("models", f"{tag}.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    # Halve the learning rate when validation loss stops improving for 2 epochs.
    # Smaller late-stage steps stop the validation loss from jumping around.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    # Stop when validation loss has not improved for 5 epochs and roll back to the
    # best weights. start_from_epoch keeps the 20 epochs the assignment requires;
    # it only matters for longer runs (e.g. python src/train.py A adam 40).
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True,
        start_from_epoch=MIN_EPOCHS, verbose=1)

    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        callbacks=[timer, lr_log, checkpoint, reduce_lr, early_stop],
        verbose=2,  # one line per epoch
    )

    # Save the numbers so the report tables can be built later by evaluate.py
    trainable_params = sum(int(tf.size(weight)) for weight in model.trainable_weights)
    results = {
        "history": {metric: [float(value) for value in values]
                    for metric, values in history.history.items()},
        "epoch_times": timer.times,
        "initial_learning_rate": lr,
        "epochs_requested": epochs,
        "epochs_run": len(timer.times),  # can be fewer than requested if EarlyStopping fired
        "momentum": MOMENTUM if opt_name == "sgd_momentum" else 0.0,
        "learning_rate_per_epoch": lr_log.lrs,
        "trainable_params": int(trainable_params),
    }
    with open(os.path.join("results", f"{tag}_history.json"), "w") as json_file:
        json.dump(results, json_file, indent=2)

    path = plot_history(history.history, tag)
    print()
    print(f"Saved curves to {path}")
    print(f"Learning rate: started at {lr:g}, ended at {lr_log.lrs[-1]:.6g}")

    # Epoch 1 is slower because it also loads every image into the cache, so skip it
    later_epoch_times = timer.times[1:]
    mean_time = sum(later_epoch_times) / len(later_epoch_times)
    print(f"Mean epoch time (epochs 2-{len(timer.times)}): {mean_time:.1f} s")


if __name__ == "__main__":
    main()
