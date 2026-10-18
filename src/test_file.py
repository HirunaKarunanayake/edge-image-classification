"""Tests for the two custom CNNs (src/modelA.py and src/modelB.py)."""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from modelA import build_model_a
from modelB import build_model_b

BUILDERS = {"A": build_model_a, "B": build_model_b}


def trainable_count(model):
    return sum(int(tf.size(w)) for w in model.trainable_weights)


@pytest.fixture(scope="module")
def models():
    """Build each model once for the whole module (building is the slow part)."""
    return {name: build() for name, build in BUILDERS.items()}


@pytest.fixture
def batch():
    """Four random 64x64 RGB images in [0, 1], like the data pipeline produces."""
    return np.random.default_rng(0).random((4, 64, 64, 3), dtype=np.float32)


@pytest.mark.parametrize("name", ["A", "B"])
def test_output_is_one_probability_per_class(models, batch, name):
    probs = models[name].predict(batch, verbose=0)
    assert probs.shape == (4, 6)
    assert np.all(probs >= 0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", ["A", "B"])
def test_input_shape_is_64x64_rgb(models, name):
    assert tuple(models[name].input_shape) == (None, 64, 64, 3)


def test_model_a_parameter_count(models):
    # 4 conv blocks + BatchNorm gamma/beta + Dense(64) + Dense(6), as in the report
    assert trainable_count(models["A"]) == 250_182


def test_model_b_parameter_count(models):
    assert trainable_count(models["B"]) == 38_113


def test_model_b_is_under_assignment_limit(models):
    assert trainable_count(models["B"]) <= 100_000


def test_model_b_is_much_smaller_than_model_a(models):
    assert trainable_count(models["A"]) / trainable_count(models["B"]) > 5


def test_model_a_uses_standard_convolutions_with_pooling(models):
    convs = [l for l in models["A"].layers if type(l) is layers.Conv2D]
    pools = [l for l in models["A"].layers if isinstance(l, layers.MaxPooling2D)]
    assert len(convs) == 4
    assert len(pools) == 4


def test_model_b_uses_only_separable_convolutions(models):
    model = models["B"]
    separable = [l for l in model.layers if isinstance(l, layers.SeparableConv2D)]
    plain = [l for l in model.layers if type(l) is layers.Conv2D]
    assert len(separable) == 4
    assert plain == []


@pytest.mark.parametrize("name", ["A", "B"])
def test_batchnorm_uses_momentum_0_9(models, name):
    bns = [l for l in models[name].layers if isinstance(l, layers.BatchNormalization)]
    assert len(bns) == 4
    assert all(abs(bn.momentum - 0.9) < 1e-9 for bn in bns)


@pytest.mark.parametrize("name, rate", [("A", 0.3), ("B", 0.2)])
def test_dropout_rate(models, name, rate):
    drops = [l for l in models[name].layers if isinstance(l, layers.Dropout)]
    assert [d.rate for d in drops] == [rate]


@pytest.mark.parametrize("name", ["A", "B"])
def test_inference_is_deterministic(models, batch, name):
    """Dropout and augmentation must be switched off at test time."""
    first = models[name](batch, training=False).numpy()
    second = models[name](batch, training=False).numpy()
    np.testing.assert_allclose(first, second, atol=1e-6)


@pytest.mark.parametrize("name", ["A", "B"])
def test_training_mode_is_random(models, batch, name):
    """In training mode augmentation and dropout should change the output."""
    first = models[name](batch, training=True).numpy()
    second = models[name](batch, training=True).numpy()
    assert not np.allclose(first, second)


@pytest.mark.parametrize("name", ["A", "B"])
def test_save_and_reload_gives_same_predictions(tmp_path, models, batch, name):
    model = models[name]
    path = tmp_path / f"{name}.keras"
    model.save(path)
    reloaded = tf.keras.models.load_model(path)
    np.testing.assert_allclose(
        model.predict(batch, verbose=0), reloaded.predict(batch, verbose=0), atol=1e-5)
