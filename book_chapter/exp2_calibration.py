"""Experiment 2: calibrated uncertainty for operational risk flagging.

Compares three uncertainty methods on SELENE predictions -- MC-Dropout, a deep
ensemble, and quantile regression -- with calibration (ECE, reliability diagram)
and prediction-interval metrics (PICP, MPIW). Writes outputs/exp2_metrics.json
and grayscale-safe figures.
"""
import json
import os
import numpy as np
import tensorflow as tf
from scipy import stats
import selene_common as sc
import plot_utils as pu
import matplotlib.pyplot as plt

OUT = pu.OUT
Z90 = 1.6448536269514722  # z for a 90% two-sided interval


def picp_mpiw(y, lo, hi):
    y = y.ravel()
    lo = lo.ravel()
    hi = hi.ravel()
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def gaussian_ece(y, mu, sigma, n_bins=10):
    """Empirical vs nominal central-interval coverage.

    Returns (mean|error|, nominal_grid, empirical_coverage_grid).
    """
    y = y.ravel()
    mu = mu.ravel()
    sigma = np.clip(sigma.ravel(), 1e-9, None)
    ps = np.linspace(0.05, 0.95, n_bins)
    cov = [float(np.mean(np.abs(y - mu) <= stats.norm.ppf(0.5 + p / 2) * sigma))
           for p in ps]
    ece = float(np.mean([abs(c - p) for c, p in zip(cov, ps)]))
    return ece, ps.tolist(), cov


def main():
    pu.gray_style()
    X, y, _ = sc.load_data()
    Xtr, Xv, ytr, yv = sc.split(X, y)
    results = {}
    curves = {}
    ps = None

    # --- MC-Dropout: keep dropout active at inference ---
    m, _ = sc.train_selene(Xtr, ytr, Xv, yv)
    preds = np.stack([m(Xv, training=True).numpy().ravel() for _ in range(100)])
    mu, sigma = preds.mean(0), preds.std(0)
    picp, mpiw = picp_mpiw(yv, mu - Z90 * sigma, mu + Z90 * sigma)
    ece, ps, cov = gaussian_ece(yv, mu, sigma)
    results["mc_dropout"] = {"picp": picp, "mpiw": mpiw, "ece": ece,
                             "rmse": float(np.sqrt(np.mean((yv.ravel() - mu) ** 2)))}
    curves["mc_dropout"] = cov

    # --- Deep ensemble: N independently trained ANNs ---
    ens = []
    for i in range(10):
        tf.random.set_seed(1000 + i)
        mi, _ = sc.train_selene(Xtr, ytr, Xv, yv)
        ens.append(mi.predict(Xv, verbose=0).ravel())
    ens = np.stack(ens)
    mu, sigma = ens.mean(0), ens.std(0)
    picp, mpiw = picp_mpiw(yv, mu - Z90 * sigma, mu + Z90 * sigma)
    ece, _, cov = gaussian_ece(yv, mu, sigma)
    results["deep_ensemble"] = {"picp": picp, "mpiw": mpiw, "ece": ece,
                                "rmse": float(np.sqrt(np.mean((yv.ravel() - mu) ** 2)))}
    curves["deep_ensemble"] = cov

    # --- Quantile regression: pinball loss at 0.05 / 0.5 / 0.95 ---
    def pinball(q):
        def loss(yt, yp):
            e = yt - yp
            return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
        return loss

    q = {}
    for qq in [0.05, 0.5, 0.95]:
        mq = sc.build_selene(18)
        mq.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=pinball(qq))
        mq.fit(Xtr, ytr, epochs=200, batch_size=16, verbose=0,
               validation_data=(Xv, yv))
        q[qq] = mq.predict(Xv, verbose=0).ravel()
    lo, med, hi = q[0.05], q[0.5], q[0.95]
    picp, mpiw = picp_mpiw(yv, lo, hi)
    sigma_q = (hi - lo) / (2 * Z90)  # symmetric proxy for the reliability curve
    ece, _, cov = gaussian_ece(yv, med, sigma_q)
    results["quantile"] = {"picp": picp, "mpiw": mpiw, "ece": ece,
                           "rmse": float(np.sqrt(np.mean((yv.ravel() - med) ** 2)))}
    curves["quantile"] = cov

    # --- Reliability diagram ---
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", label="ideal")
    for name in ["mc_dropout", "deep_ensemble", "quantile"]:
        ax.plot(ps, curves[name], marker="o", label=name.replace("_", " "))
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.legend()
    pu.savefig(fig, "fig_reliability")

    # --- Calibration vs sharpness (ECE vs MPIW) ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    markers = {"mc_dropout": "o", "deep_ensemble": "s", "quantile": "^"}
    for name in ["mc_dropout", "deep_ensemble", "quantile"]:
        ax.scatter(results[name]["mpiw"], results[name]["ece"], s=90,
                   marker=markers[name], edgecolor="black",
                   label=name.replace("_", " "))
    ax.set_xlabel("Mean interval width (sharpness)")
    ax.set_ylabel("Calibration error (ECE)")
    ax.legend()
    pu.savefig(fig, "fig_calibration_sharpness")

    with open(os.path.join(OUT, "exp2_metrics.json"), "w") as f:
        json.dump({"results": results, "nominal": ps}, f, indent=2)
    print("Exp2 done:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
