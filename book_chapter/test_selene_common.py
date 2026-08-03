"""Smoke tests for selene_common: shapes, feature count, metrics."""
import numpy as np
import selene_common as sc


def test_features_count():
    assert len(sc.FEATURES) == 18


def test_load_and_split_shapes():
    X, y, scaler = sc.load_data()
    assert X.shape[1] == 18
    assert X.shape[0] == y.shape[0] > 1000
    Xtr, Xv, ytr, yv = sc.split(X, y)
    assert Xtr.shape[0] > Xv.shape[0]
    assert abs(Xv.shape[0] / X.shape[0] - 0.1) < 0.02


def test_build_selene_output_shape():
    m = sc.build_selene(18)
    out = m(np.zeros((3, 18), dtype="float32"))
    assert out.shape == (3, 1)


def test_metrics_perfect():
    y = np.array([[1.0], [2.0], [3.0]])
    d = sc.metrics(y, y)
    assert d["r2"] > 0.999 and d["mae"] < 1e-9
