"""Experiment 1: physical interpretability and ablation of SELENE.

Produces permutation importance, a top-k feature ablation curve, SHAP
attributions, and an XGBoost gain-importance cross-check. Writes real metrics to
outputs/exp1_metrics.json and grayscale-safe figures.
"""
import json
import os
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import selene_common as sc
import plot_utils as pu

OUT = pu.OUT


class KerasReg(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper so permutation_importance can score the ANN."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).ravel()


def main():
    pu.gray_style()
    X, y, _ = sc.load_data()
    Xtr, Xv, ytr, yv = sc.split(X, y)
    model, _ = sc.train_selene(Xtr, ytr, Xv, yv)
    base = sc.metrics(yv, model.predict(Xv, verbose=0))

    # --- Permutation importance (higher = more important) ---
    reg = KerasReg(model)
    pi = permutation_importance(reg, Xv, yv.ravel(), n_repeats=20,
                                random_state=42, scoring="r2")
    order = np.argsort(pi.importances_mean)[::-1]
    feats = [sc.FEATURES[i] for i in order]
    means = pi.importances_mean[order]
    stds = pi.importances_std[order]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(feats)), means, xerr=stds, color="0.4", edgecolor="black")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (drop in R^2)")
    pu.savefig(fig, "fig_permutation_importance")

    # --- Ablation: retrain on top-k permutation features ---
    ablation = []
    for k in [2, 4, 8, 12, 18]:
        idx = order[:k]
        mk, _ = sc.train_selene(Xtr[:, idx], ytr, Xv[:, idx], yv)
        ablation.append({"k": int(k),
                         **sc.metrics(yv, mk.predict(Xv[:, idx], verbose=0))})
    ks = [a["k"] for a in ablation]
    r2s = [a["r2"] for a in ablation]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, r2s, marker="o", color="black")
    ax.set_xlabel("Number of features (top-k)")
    ax.set_ylabel("Validation R^2")
    pu.savefig(fig, "fig_ablation_curve")

    # --- SHAP (KernelExplainer on a background sample) ---
    bg = shap.sample(Xtr, 100, random_state=42)
    expl = shap.KernelExplainer(
        lambda d: model.predict(d, verbose=0).ravel(), bg)
    sv = expl.shap_values(Xv[:200], nsamples=100)
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, outputs) -> squeeze single output
        sv = sv[..., 0]
    shap_mean_abs = np.abs(sv).mean(axis=0)
    sorder = np.argsort(shap_mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(18), shap_mean_abs[sorder], color="0.5", edgecolor="black",
            hatch="...")
    ax.set_yticks(range(18))
    ax.set_yticklabels([sc.FEATURES[i] for i in sorder])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    pu.savefig(fig, "fig_shap_summary")

    # --- XGBoost gain-based importance cross-check ---
    dm = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                          random_state=42)
    dm.fit(Xtr, ytr.ravel())
    xgb_imp = dm.feature_importances_
    xorder = np.argsort(xgb_imp)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(18), xgb_imp[xorder], color="0.6", edgecolor="black",
            hatch="///")
    ax.set_yticks(range(18))
    ax.set_yticklabels([sc.FEATURES[i] for i in xorder])
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost gain importance")
    pu.savefig(fig, "fig_xgb_importance")

    out = {
        "base_metrics": base,
        "permutation_importance": {f: float(m) for f, m in zip(feats, means)},
        "ablation": ablation,
        "shap_mean_abs": {sc.FEATURES[i]: float(shap_mean_abs[i])
                          for i in range(18)},
        "xgb_importance": {sc.FEATURES[i]: float(xgb_imp[i]) for i in range(18)},
    }
    with open(os.path.join(OUT, "exp1_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Exp1 done. Base:", base)


if __name__ == "__main__":
    main()
