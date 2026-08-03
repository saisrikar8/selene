# Lunar Radiation Book Chapter (SELENE Extension) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Springer LaTeX book chapter that extends the published SELENE paper with two new experiments (physical interpretability/ablation and calibrated uncertainty), plus expanded background/related-work, meeting all publisher constraints.

**Architecture:** A small reproducible Python package (`book_chapter/`) rebuilds the SELENE ANN from the existing training code, then runs two analysis experiments that emit real metrics (JSON) and grayscale-safe figures (300 dpi PNG + PDF). A new Springer LaTeX project (`Lunar_Research_Book_Chapter/`) consumes those artifacts and prose sections. Reproduction is deterministic (fixed seeds) so numbers are stable across reruns.

**Tech Stack:** Python 3.11+ (venv), numpy, pandas, scikit-learn, scipy, xgboost, tensorflow (Apple-Silicon `tensorflow` wheel), shap, matplotlib; LaTeX (Springer `svmono`/`llncs`).

## Global Constraints

- Manuscript length: **15–25 pages**, single column, Springer book-chapter template.
- **≥50% new material** vs. published paper; **different title and abstract**; **cite the original SELENE paper**.
- Authors: **Sai Srikar Tummala, Mittansh Bhatia, Robert Chun** — same three, same order.
- All figures **grayscale-safe** (linestyle/marker/hatch, not color alone), **300 dpi**, exported as **separate files** and embedded.
- **No orphan citations** in either direction.
- Base data file: `aligned_averaged_by_crater.csv`; target column `dose_D1`; 18 features (exact list in Task 1).
- Determinism: set `PYTHONHASHSEED=0`, `np.random.seed(42)`, `tf.random.set_seed(42)` in every script.
- Git: repo is shared `surentech-org/surentech-server` on `main`. **Do NOT push. Do NOT commit to `main`.** Create branch `book-chapter/selene-extension` before any commit; commits are local until the author approves.
- All new code lives under `lunar-radiation/book_chapter/`; all outputs under `lunar-radiation/book_chapter/outputs/`.

---

### Task 0: Environment & reproducibility harness

**Files:**
- Create: `book_chapter/requirements.txt`
- Create: `book_chapter/README.md`
- Create branch: `book-chapter/selene-extension`

**Interfaces:**
- Produces: a working `.venv` at `lunar-radiation/.venv` with all deps importable.

- [ ] **Step 1: Create the working branch**

```bash
cd /Users/sst/Desktop/work/research/lunar-radiation
git checkout -b book-chapter/selene-extension
```

- [ ] **Step 2: Write `book_chapter/requirements.txt`**

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
scipy==1.13.1
xgboost==2.1.1
tensorflow==2.16.2
shap==0.46.0
matplotlib==3.9.1
joblib==1.4.2
```

- [ ] **Step 3: Create venv and install**

```bash
cd /Users/sst/Desktop/work/research/lunar-radiation
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r book_chapter/requirements.txt
```
Expected: all install without error. If the pinned `tensorflow` wheel is unavailable for this Python/arch, relax to the newest `tensorflow` 2.x that installs, and record the actual version in `book_chapter/README.md`.

- [ ] **Step 4: Verify imports and data availability**

```bash
cd /Users/sst/Desktop/work/research/lunar-radiation
./.venv/bin/python -c "import numpy,pandas,sklearn,scipy,xgboost,tensorflow,shap,matplotlib,joblib; print('imports OK')"
./.venv/bin/python -c "import pandas as pd; c=pd.read_csv('aligned_averaged_by_crater.csv', nrows=5); print('dose_D1' in c.columns, len(c.columns))"
```
Expected: `imports OK`, then `True <int>`.

- [ ] **Step 5: Write `book_chapter/README.md`** documenting: venv path, actual installed versions, how to run each experiment, and the determinism note.

- [ ] **Step 6: Commit**

```bash
git add book_chapter/requirements.txt book_chapter/README.md
git commit -m "chore(book-chapter): environment and reproducibility harness"
```

---

### Task 1: Shared SELENE module (data + model, non-interactive)

**Files:**
- Create: `book_chapter/selene_common.py`
- Create: `book_chapter/test_selene_common.py`
- Reference (do not modify): `train_model.py:20-92`

**Interfaces:**
- Produces:
  - `FEATURES: list[str]` — the 18 feature column names.
  - `load_data() -> tuple[np.ndarray, np.ndarray, StandardScaler]` — returns `(X_scaled, y, scaler_X)` from `aligned_averaged_by_crater.csv`, dropping NaN targets and rows with any negative value in features or target.
  - `split(X, y) -> tuple` — deterministic 90/10 shuffle split (`random_state=42`), returns `(X_train, X_val, y_train, y_val)`.
  - `build_selene(input_dim: int, dropout1=0.4, dropout2=0.3) -> tf.keras.Model` — the published architecture.
  - `train_selene(X_train, y_train, X_val, y_val, lr=1e-3, epochs=300) -> tuple[Model, dict]` — trains with EarlyStopping + ReduceLROnPlateau, returns `(model, history_dict)`.
  - `metrics(y_true, y_pred) -> dict` — keys `mse, mae, r2, mape`.

- [ ] **Step 1: Write the failing test**

```python
# book_chapter/test_selene_common.py
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
    y = np.array([[1.0],[2.0],[3.0]])
    d = sc.metrics(y, y)
    assert d["r2"] > 0.999 and d["mae"] < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd book_chapter && ../.venv/bin/python -m pytest test_selene_common.py -v`
Expected: FAIL (`No module named 'selene_common'`).

- [ ] **Step 3: Write `book_chapter/selene_common.py`**

```python
import os, numpy as np, pandas as pd, tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42); tf.random.set_seed(42)

CSV = os.path.join(os.path.dirname(__file__), "..", "aligned_averaged_by_crater.csv")
FEATURES = [
    "sis_cnt_O_1", "sis_cnt_O_2", "sis_cnt_C_2",
    "ace_cnt_Fe_4", "ace_cnt_Ni_2", "ace_cnt_Cr_4",
    "ace_cnt_S_3", "ace_cnt_Ti_4", "ace_cnt_Fe_7",
    "ace_cnt_Ni_1", "ace_cnt_F_3", "ace_cnt_Ca_5",
    "ace_cnt_S_4", "ace_cnt_Fe_5", "ace_cnt_Ar_5",
    "ace_cnt_Ne_2", "ace_cnt_Si_6", "ace_cnt_B_5",
]

def load_data():
    df = pd.read_csv(CSV).dropna(subset=["dose_D1"])
    cols = FEATURES + ["dose_D1"]
    df = df[(df[cols] >= 0).all(axis=1)]
    y = df["dose_D1"].to_numpy().reshape(-1, 1)
    X = df[FEATURES].astype(float).to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

def split(X, y):
    Xs, ys = shuffle(X, y, random_state=42)
    i = int(len(Xs) * 0.9)
    return Xs[:i], Xs[i:], ys[:i], ys[i:]

def build_selene(input_dim, dropout1=0.4, dropout2=0.3):
    return Sequential([
        Dense(64, kernel_regularizer=l2(1e-2), input_dim=input_dim),
        BatchNormalization(), LeakyReLU(alpha=0.01), Dropout(dropout1),
        Dense(32, kernel_regularizer=l2(1e-2)),
        BatchNormalization(), LeakyReLU(alpha=0.01), Dropout(dropout2),
        Dense(1),
    ])

def train_selene(X_train, y_train, X_val, y_val, lr=1e-3, epochs=300):
    m = build_selene(X_train.shape[1])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae",
              metrics=["mae", "mape", "mse"])
    cb = [EarlyStopping(patience=25, restore_best_weights=True),
          ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-10, verbose=0)]
    h = m.fit(X_train, y_train, epochs=epochs, batch_size=16,
              validation_data=(X_val, y_val), callbacks=cb, verbose=0)
    return m, h.history

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) /
                     np.clip(np.abs(y_true), 1e-12, None))) * 100),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd book_chapter && ../.venv/bin/python -m pytest test_selene_common.py -v`
Expected: 4 passed.

- [ ] **Step 5: Sanity-check reproduction quality**

Run:
```bash
cd book_chapter && ../.venv/bin/python -c "
import selene_common as sc
X,y,_=sc.load_data(); Xtr,Xv,ytr,yv=sc.split(X,y)
m,_=sc.train_selene(Xtr,ytr,Xv,yv)
print(sc.metrics(yv, m.predict(Xv,verbose=0)))"
```
Expected: an `r2` in a comparable ballpark to the paper (roughly 0.7–0.9). Record the actual reproduced metrics in `book_chapter/README.md` (they are the numbers the chapter's recap will cite as "reproduced", noting run-to-run variation). If `r2` is far below 0.5, stop and report — do not proceed to experiments on a broken base model.

- [ ] **Step 6: Commit**

```bash
git add book_chapter/selene_common.py book_chapter/test_selene_common.py book_chapter/README.md
git commit -m "feat(book-chapter): reproducible SELENE data+model module"
```

---

### Task 2: Experiment 1 — Physical interpretability & ablation

**Files:**
- Create: `book_chapter/exp1_interpretability.py`
- Create: `book_chapter/plot_utils.py`
- Output: `book_chapter/outputs/exp1_metrics.json`, `outputs/fig_permutation_importance.{png,pdf}`, `outputs/fig_ablation_curve.{png,pdf}`, `outputs/fig_shap_summary.{png,pdf}`, `outputs/fig_xgb_importance.{png,pdf}`

**Interfaces:**
- Consumes: `selene_common.{load_data,split,train_selene,build_selene,metrics,FEATURES}`.
- Produces: `plot_utils.gray_style()` — sets a grayscale-safe matplotlib rcParams; `plot_utils.savefig(fig, name)` — writes `outputs/<name>.png` (300 dpi) and `<name>.pdf`; `plot_utils.OUT` — outputs dir path.

- [ ] **Step 1: Write `book_chapter/plot_utils.py`**

```python
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "ooo"]

def gray_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
        "image.cmap": "gray", "axes.prop_cycle":
            plt.cycler(color=["0.1","0.4","0.6","0.75"]) +
            plt.cycler(linestyle=["-","--","-.",":"]),
    })

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
```

- [ ] **Step 2: Write `book_chapter/exp1_interpretability.py`**

```python
import json, os, numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import selene_common as sc
import plot_utils as pu

OUT = pu.OUT

class KerasReg(BaseEstimator, RegressorMixin):
    """sklearn wrapper so permutation_importance can score the ANN."""
    def __init__(self, model): self.model = model
    def fit(self, X, y): return self
    def predict(self, X): return self.model.predict(X, verbose=0).ravel()

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
    means = pi.importances_mean[order]; stds = pi.importances_std[order]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(feats)), means, xerr=stds, color="0.4",
            edgecolor="black")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats)
    ax.invert_yaxis(); ax.set_xlabel("Permutation importance (drop in R^2)")
    pu.savefig(fig, "fig_permutation_importance")

    # --- Ablation: retrain on top-k permutation features ---
    ablation = []
    for k in [2, 4, 8, 12, 18]:
        idx = order[:k]
        mk, _ = sc.train_selene(Xtr[:, idx], ytr, Xv[:, idx], yv)
        ablation.append({"k": int(k),
                         **sc.metrics(yv, mk.predict(Xv[:, idx], verbose=0))})
    ks = [a["k"] for a in ablation]; r2s = [a["r2"] for a in ablation]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, r2s, marker="o", color="black")
    ax.set_xlabel("Number of features (top-k)"); ax.set_ylabel("Validation R^2")
    pu.savefig(fig, "fig_ablation_curve")

    # --- SHAP (KernelExplainer on a background sample) ---
    bg = shap.sample(Xtr, 100, random_state=42)
    expl = shap.KernelExplainer(lambda d: model.predict(d, verbose=0).ravel(), bg)
    sv = expl.shap_values(Xv[:200], nsamples=100)
    fig = plt.figure(figsize=(7, 6))
    shap.summary_plot(sv, Xv[:200], feature_names=sc.FEATURES,
                      show=False, color_bar=False, cmap="gray")
    pu.savefig(fig, "fig_shap_summary")
    shap_mean_abs = np.abs(sv).mean(axis=0)

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
    ax.invert_yaxis(); ax.set_xlabel("XGBoost gain importance")
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
```

- [ ] **Step 3: Run the experiment**

Run: `cd book_chapter && ../.venv/bin/python exp1_interpretability.py`
Expected: prints `Exp1 done. Base: {...}`; creates `outputs/exp1_metrics.json` and the four figure pairs.

- [ ] **Step 4: Verify outputs exist and are non-trivial**

```bash
cd book_chapter && ../.venv/bin/python -c "
import json; d=json.load(open('outputs/exp1_metrics.json'))
assert d['ablation'][-1]['k']==18
assert len(d['permutation_importance'])==18
print('top-3 by permutation:', list(d['permutation_importance'])[:3])"
ls -la book_chapter/outputs/fig_permutation_importance.png
```
Expected: assertion passes, prints 3 feature names, PNG exists.

- [ ] **Step 5: Commit**

```bash
git add book_chapter/plot_utils.py book_chapter/exp1_interpretability.py book_chapter/outputs
git commit -m "feat(book-chapter): experiment 1 interpretability and ablation"
```

---

### Task 3: Experiment 2 — Calibrated uncertainty

**Files:**
- Create: `book_chapter/exp2_calibration.py`
- Output: `book_chapter/outputs/exp2_metrics.json`, `outputs/fig_reliability.{png,pdf}`, `outputs/fig_calibration_sharpness.{png,pdf}`

**Interfaces:**
- Consumes: `selene_common.*`, `plot_utils.*`.
- Produces: `exp2_metrics.json` with `results` = dict of `mc_dropout`, `deep_ensemble`, `quantile`, each `{picp, mpiw, ece, rmse}`, plus `nominal` coverage grid.

- [ ] **Step 1: Write `book_chapter/exp2_calibration.py`**

```python
import json, os, numpy as np, tensorflow as tf
from scipy import stats
import selene_common as sc
import plot_utils as pu
import matplotlib.pyplot as plt

OUT = pu.OUT
Z90 = 1.6448536269514722  # z for 90% two-sided interval

def picp_mpiw(y, lo, hi):
    y = y.ravel(); lo = lo.ravel(); hi = hi.ravel()
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))

def gaussian_ece(y, mu, sigma, n_bins=10):
    """Empirical vs nominal central-interval coverage. Returns
    (mean|error|, nominal_grid, empirical_coverage_grid)."""
    y = y.ravel(); mu = mu.ravel(); sigma = np.clip(sigma.ravel(), 1e-9, None)
    ps = np.linspace(0.05, 0.95, n_bins)
    cov = [float(np.mean(np.abs(y - mu) <= stats.norm.ppf(0.5 + p/2) * sigma))
           for p in ps]
    ece = float(np.mean([abs(c - p) for c, p in zip(cov, ps)]))
    return ece, ps.tolist(), cov

def main():
    pu.gray_style()
    X, y, _ = sc.load_data()
    Xtr, Xv, ytr, yv = sc.split(X, y)
    results = {}; curves = {}; ps = None

    # --- MC-Dropout: keep dropout active at inference ---
    m, _ = sc.train_selene(Xtr, ytr, Xv, yv)
    preds = np.stack([m(Xv, training=True).numpy().ravel() for _ in range(100)])
    mu, sigma = preds.mean(0), preds.std(0)
    picp, mpiw = picp_mpiw(yv, mu - Z90*sigma, mu + Z90*sigma)
    ece, ps, cov = gaussian_ece(yv, mu, sigma)
    results["mc_dropout"] = {"picp": picp, "mpiw": mpiw, "ece": ece,
                             "rmse": float(np.sqrt(np.mean((yv.ravel()-mu)**2)))}
    curves["mc_dropout"] = cov

    # --- Deep ensemble: N independently trained ANNs ---
    ens = []
    for i in range(10):
        tf.random.set_seed(1000 + i)
        mi, _ = sc.train_selene(Xtr, ytr, Xv, yv)
        ens.append(mi.predict(Xv, verbose=0).ravel())
    ens = np.stack(ens); mu, sigma = ens.mean(0), ens.std(0)
    picp, mpiw = picp_mpiw(yv, mu - Z90*sigma, mu + Z90*sigma)
    ece, _, cov = gaussian_ece(yv, mu, sigma)
    results["deep_ensemble"] = {"picp": picp, "mpiw": mpiw, "ece": ece,
                                "rmse": float(np.sqrt(np.mean((yv.ravel()-mu)**2)))}
    curves["deep_ensemble"] = cov

    # --- Quantile regression: pinball loss at 0.05/0.5/0.95 ---
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
    sigma_q = (hi - lo) / (2 * Z90)  # symmetric proxy for reliability curve
    ece, _, cov = gaussian_ece(yv, med, sigma_q)
    results["quantile"] = {"picp": picp, "mpiw": mpiw, "ece": ece,
                           "rmse": float(np.sqrt(np.mean((yv.ravel()-med)**2)))}
    curves["quantile"] = cov

    # --- Reliability diagram ---
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", label="ideal")
    for name in ["mc_dropout", "deep_ensemble", "quantile"]:
        ax.plot(ps, curves[name], marker="o", label=name.replace("_", " "))
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.legend(); pu.savefig(fig, "fig_reliability")

    # --- Calibration vs sharpness (ECE vs MPIW) ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    markers = {"mc_dropout": "o", "deep_ensemble": "s", "quantile": "^"}
    for name in ["mc_dropout", "deep_ensemble", "quantile"]:
        ax.scatter(results[name]["mpiw"], results[name]["ece"], s=90,
                   marker=markers[name], edgecolor="black",
                   label=name.replace("_", " "))
    ax.set_xlabel("Mean interval width (sharpness)")
    ax.set_ylabel("Calibration error (ECE)"); ax.legend()
    pu.savefig(fig, "fig_calibration_sharpness")

    with open(os.path.join(OUT, "exp2_metrics.json"), "w") as f:
        json.dump({"results": results, "nominal": ps}, f, indent=2)
    print("Exp2 done:", json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run: `cd book_chapter && ../.venv/bin/python exp2_calibration.py`
Expected: prints per-method `{picp, mpiw, ece, rmse}`; writes `exp2_metrics.json` and both figures. (Deep ensemble trains 10 models — allow several minutes on CPU.)

- [ ] **Step 3: Verify outputs**

```bash
cd book_chapter && ../.venv/bin/python -c "
import json; d=json.load(open('outputs/exp2_metrics.json'))['results']
for k,v in d.items(): assert 0<=v['picp']<=1 and v['mpiw']>0
print({k: round(v['picp'],3) for k,v in d.items()})"
```
Expected: prints PICP per method (each in [0,1]).

- [ ] **Step 4: Commit**

```bash
git add book_chapter/exp2_calibration.py book_chapter/outputs
git commit -m "feat(book-chapter): experiment 2 calibrated uncertainty"
```

---

### Task 4: Springer LaTeX scaffold + bibliography

**Files:**
- Create: `Lunar_Research_Book_Chapter/main.tex` (skeleton: title, authors, abstract, empty sections)
- Create: `Lunar_Research_Book_Chapter/references.bib` (existing bib + SELENE self-citation)
- Copy: `book_chapter/outputs/*.pdf` → `Lunar_Research_Book_Chapter/figures/`

**Interfaces:**
- Produces: a document that compiles to a title page + abstract + section headings.

- [ ] **Step 1: Create the Springer skeleton `main.tex`**

Use the Springer `svmono`/`llncs` class if the template zip is available locally; otherwise fall back to a single-column `article` and add a comment noting the template swap. Include: new title (spec §3), the three authors in order with published affiliations/emails, the new abstract (spec §3), and empty `\section` headings matching outline §4. Add `\bibliographystyle{spmpsci}` (or `IEEEtran` fallback) and `\bibliography{references}`.

- [ ] **Step 2: Build `references.bib`**

Copy every entry from `Lunar_Research/references.bib`, then add the self-citation with author-confirmable fields marked:

```bibtex
@inproceedings{tummala2025selene,
  author    = {Tummala, Sai Srikar and Bhatia, Mittansh and Chun, Robert},
  title     = {{SELENE}: A Deep Learning Model for Predicting Lunar Absorbed Radiation Dose Rates},
  booktitle = {Proceedings of the [VENUE -- AUTHOR TO CONFIRM: CSCE or ICBAIE]},
  year      = {2025},
  note      = {Original conference paper extended by this chapter}
}
```

- [ ] **Step 3: Copy figures**

```bash
cd /Users/sst/Desktop/work/research/lunar-radiation
mkdir -p Lunar_Research_Book_Chapter/figures
cp book_chapter/outputs/*.pdf Lunar_Research_Book_Chapter/figures/
```

- [ ] **Step 4: Compile the skeleton**

```bash
cd Lunar_Research_Book_Chapter && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```
Expected: `main.pdf` produced. If the Springer class is missing, switch to the `article` fallback and record it in the chapter README.

- [ ] **Step 5: Commit**

```bash
git add Lunar_Research_Book_Chapter
git commit -m "feat(book-chapter): Springer LaTeX scaffold and bibliography"
```

---

### Task 5: Write the prose sections

**Files:**
- Modify: `Lunar_Research_Book_Chapter/main.tex` (fill sections 1–4, 7–8 from outline)

**Interfaces:** none (prose). Every claim recapped from the original paper must cite `tummala2025selene`.

- [ ] **Step 1: Write Introduction** [EXPANDED] — Artemis motivation; the trustworthiness thesis; explicitly state the two new contributions; cite `tummala2025selene`, `Spence2010`, `Looper2020`.
- [ ] **Step 2: Write Background** [NEW] — GCR vs SEP; absorbed dose / LET / dose-equivalent / quality-factor definitions with equations; unshielded Moon; CRaTER + ACE SIS/CRIS instrumentation. Cite `Spence2010, Stone1998ACE, Stone1998SIS, Cummings1998, Zhang2020, Wimmer2020`.
- [ ] **Step 3: Write Related Work** [EXPANDED] — physics models (`2022cosp...44.2697D, Du2024, Singleterry2010, Slaba2013`), ML forecasting (`Papagiannopoulos2023, Zampieri2024`), and a new UQ/interpretability-in-scientific-ML subsection (cite `somepalli2021saint, chen2016xgboost, breiman2001random, bishop2006pattern`).
- [ ] **Step 4: Write SELENE Recap** [RECAP] — condensed data pipeline + architecture + headline results, all attributed to `tummala2025selene`; note reproduced metrics from Task 1 Step 5.
- [ ] **Step 5: Write Discussion + Future Work + Conclusion** [NEW/EXPANDED] — leave two clearly-marked placeholders that Task 6 fills with actual experiment numbers; everything else complete prose.
- [ ] **Step 6: Compile and eyeball**

Run: `cd Lunar_Research_Book_Chapter && pdflatex -interaction=nonstopmode main.tex >/dev/null && echo OK`
Expected: `OK`, no undefined-citation errors for cited keys.

- [ ] **Step 7: Commit**

```bash
git add Lunar_Research_Book_Chapter/main.tex
git commit -m "docs(book-chapter): background, related work, recap, discussion prose"
```

---

### Task 6: Integrate experiment results (sections 5 & 6)

**Files:**
- Modify: `Lunar_Research_Book_Chapter/main.tex` (fill Experiment 1 & 2 sections + results tables)

**Interfaces:**
- Consumes: `book_chapter/outputs/exp1_metrics.json`, `exp2_metrics.json`, and the figures in `figures/`.

- [ ] **Step 1: Write Section 5 (Interpretability & Ablation)** — describe method; insert `fig_permutation_importance`, `fig_ablation_curve`, `fig_shap_summary`, `fig_xgb_importance`; build a top-features table from `exp1_metrics.json`; give the physics interpretation of the dominant ion/energy-band features. Use the **actual numbers** from the JSON (read the file; do not invent).
- [ ] **Step 2: Write Section 6 (Calibrated Uncertainty)** — describe MC-Dropout/ensemble/quantile; insert `fig_reliability`, `fig_calibration_sharpness`; build a table of `picp/mpiw/ece/rmse` per method from `exp2_metrics.json`; state which method wins the calibration/sharpness trade-off and the recommended operational risk-flag threshold.
- [ ] **Step 3: Backfill Discussion placeholders** from Task 5 Step 5 with the concrete findings.
- [ ] **Step 4: Compile**

Run: `cd Lunar_Research_Book_Chapter && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex && echo OK`
Expected: `OK`; all figures render; all table numbers match the JSON.

- [ ] **Step 5: Commit**

```bash
git add Lunar_Research_Book_Chapter/main.tex
git commit -m "docs(book-chapter): integrate experiment results, figures, and tables"
```

---

### Task 7: Finalization — reference hygiene, grayscale, page count

**Files:**
- Modify: `Lunar_Research_Book_Chapter/main.tex` as needed.
- Create: `book_chapter/check_finalization.py`

**Interfaces:** none.

- [ ] **Step 1: Write `book_chapter/check_finalization.py`** — parse `main.tex` for `\cite{...}` keys and `references.bib` for entry keys; assert no orphans in either direction; print mismatches.

```python
import re, os, sys
base = os.path.join(os.path.dirname(__file__), "..", "Lunar_Research_Book_Chapter")
tex = open(os.path.join(base, "main.tex")).read()
bib = open(os.path.join(base, "references.bib")).read()
cited = set(k.strip() for m in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex)
            for k in m.split(","))
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
orphan_bib = defined - cited
orphan_cite = cited - defined
print("Uncited bib entries:", sorted(orphan_bib))
print("Undefined citations:", sorted(orphan_cite))
sys.exit(1 if (orphan_bib or orphan_cite) else 0)
```

- [ ] **Step 2: Run the check and fix orphans**

Run: `cd book_chapter && python check_finalization.py`
Expected: exit 0. If uncited bib entries exist, either cite them in text or remove them; if undefined citations exist, add bib entries. Re-run until exit 0.

- [ ] **Step 3: Grayscale audit** — open each figure PDF in `figures/` and confirm no information is encoded by color alone. Confirm the reliability diagram's three curves are distinguishable by marker/linestyle.

- [ ] **Step 4: Page-count check**

```bash
cd Lunar_Research_Book_Chapter && pdfinfo main.pdf | grep Pages
```
Expected: Pages in **15–25**. If under 15, expand Background/Related Work; if over 25, tighten the recap.

- [ ] **Step 5: Final compile + verify the four publisher rules**

Confirm in the compiled PDF: (a) title differs from original; (b) abstract differs from original; (c) `tummala2025selene` appears in the reference list and is cited in text; (d) authors are Tummala, Bhatia, Chun in order.

- [ ] **Step 6: Commit**

```bash
git add book_chapter/check_finalization.py Lunar_Research_Book_Chapter/main.tex
git commit -m "chore(book-chapter): reference hygiene, grayscale audit, page-count finalization"
```

---

## Notes for the executor

- **Author inputs still open** (do not block; leave the marked placeholders): exact venue/year/DOI for `tummala2025selene`; whether to swap Gmail addresses for professional ones.
- **Never push; never commit to `main`.** Work stays on `book-chapter/selene-extension`.
- If reproduced SELENE metrics differ from the paper's, report the **actual** reproduced numbers — do not copy the paper's numbers as if freshly produced. The chapter should say results are "consistent with the originally reported values" and cite the paper for the originals.
