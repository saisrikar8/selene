# SELENE — Space Environment Lunar Exposure Neural Estimator

A deep learning model that predicts lunar surface radiation dose rates from space weather particle flux data, replacing slow physics simulations (GEANT4/HZETRN) with a fast neural network.

Research paper: *SELENE: A Deep Learning Model for Predicting Lunar Absorbed Radiation Dose Rates*, by Tummala, Bhatia, Chun. See [`Lunar_Research/main.tex`](Lunar_Research/main.tex).

An extended book chapter with two follow-on experiments (feature interpretability + calibrated uncertainty) lives in [`book_chapter/`](book_chapter) and [`Lunar_Research_Book_Chapter/`](Lunar_Research_Book_Chapter). See that folder's own `RUNBOOK.md`.

## What it does

SELENE is a fully connected ANN (18 input features → 64 → 32 → 1, batch norm + LeakyReLU + dropout, L2-regularized) trained on 27,000+ hours of synchronized data from:
- **CRaTER** (Cosmic Ray Telescope for the Effects of Radiation) aboard NASA's Lunar Reconnaissance Orbiter (target: daily absorbed dose rate at the lunar surface)
- **ACE** (Advanced Composition Explorer) SIS and CRIS instruments at the Earth–Sun L1 point (inputs: particle flux measurements for GCRs and SEPs)

It's benchmarked against Linear Regression, XGBoost, and a SAINT transformer, and extended with Monte Carlo Dropout for predictive uncertainty.

| Model | MSE | MAE | R² | MAPE |
|---|---|---|---|---|
| **SELENE (ANN)** | 0.00093602 | 0.00429024 | **0.8342** | **14.26%** |
| Monte Carlo Dropout | 0.00097653 | 0.00441782 | 0.8270 | 14.93% |
| XGBoost | 0.00558249 | 0.00815728 | 0.0112 | 42.83% |
| Linear Regression | 0.000416 | 0.008782 | 0.3875 | 60.67% |
| SAINT | 1.4503 | 0.8853 | −2135.48 | 79.27% |

## Repo layout

This repo accumulated a lot of iteration during development — renamed CSVs, superseded model checkpoints, and other leftover script versions. It's now organized by pipeline stage, with dead ends moved to `archive/` rather than deleted:

| Folder | Contents |
|---|---|
| `pipeline/` | Data acquisition + preprocessing: `download_sis_data.py`, `download_epam_data_async.py`, `download_goes_data.py` (raw ACE/GOES downloads), `organize_data.py` (merges dose rates + particle flux by timestamp), `avg_data.py` (aggregates to daily resolution, the temporal-alignment fix described in the paper, Section III-B) |
| `models/` | `train_model.py` — **the published SELENE ANN** (18 features, 64→32→1, batch norm + LeakyReLU + dropout); `montecarlotrain.py` — MC Dropout variant for uncertainty quantification; `train_xgboost.py`, `train_SAINT.py`, `linregtest.py` — baseline comparison models; `activations.py`, `losses.py` — shared custom Keras layers/losses |
| `evaluation/` | `evaluate_model.py` — the comprehensive evaluation script that produces every results figure in the paper (predicted-vs-true, uncertainty correlation, precision/recall/F1 vs. uncertainty threshold, outlier examples); `error_analysis.py` — error-vs-dose-magnitude analysis (Figures 11–12); `data_analysis.py` — correlation heatmaps used for feature selection (Figures 1–3) |
| `archive/` | Superseded scripts and model checkpoints from earlier iterations, kept for reference: an alternate/incomplete `evaluate_model_v2.py`, an unfinished `organize_data_2.py`, exploratory scripts (`peek_dat.py`, `preload_data.py`, `polyreg.py`, `test_saint.py`), and six early-iteration `.keras` checkpoints that predate the published `huber_lunar_radiation_model.keras` |
| `book_chapter/` | Reproducible rebuild (`selene_common.py`) plus two new experiments (interpretability, calibration) for the extended book chapter |
| `Lunar_Research/`, `Lunar_Research_Book_Chapter/` | LaTeX source for the conference paper and the book chapter |

Root-level files are the data caches and trained checkpoints the scripts above read/write: `X.npy`/`y.npy`/`X_val.npy`/`y_val.npy` (scaled feature arrays), `scaler_X.pkl`/`scaler_y.pkl`, `huber_lunar_radiation_model.keras` (published SELENE weights), `huber_lunar_radiation_model_heteroscedastic_mc_dropout.keras` (MC Dropout weights), `SAINT_model.keras`/`SAINT_model/`, and the correlation-heatmap / outlier-example figures referenced above.

## Setup

```bash
pip install numpy pandas tensorflow scikit-learn xgboost matplotlib seaborn scipy joblib requests aiohttp aiofiles beautifulsoup4 cdflib netCDF4 tqdm
```

## Running the pipeline

```bash
cd pipeline
python download_sis_data.py
python download_epam_data_async.py
python organize_data.py
python avg_data.py

cd ../models
python train_model.py          # trains SELENE, saves ../huber_lunar_radiation_model.keras
python montecarlotrain.py      # MC Dropout variant
python train_xgboost.py
python train_SAINT.py
python linregtest.py

cd ../evaluation
python evaluate_model.py       # regenerates all results figures
python data_analysis.py
python error_analysis.py
```

## Known gaps

This repo reflects real research iteration, not a polished package. A few things to know before trying to reproduce end-to-end:
- **Raw/intermediate data isn't included** (too large for this checkout). `pipeline/organize_data.py` writes `merged_lunar_radiation.csv`, but `pipeline/avg_data.py` expects `aligned_and_expanded_data.csv` as input — these don't currently match up; reconciling that filename is the main blocker to a from-scratch rerun. The cached feature arrays (`X.npy`, `y.npy`, etc.) and trained checkpoints are present, so `evaluation/evaluate_model.py` and the model-training scripts can run without redoing the data pipeline.
- `evaluation/error_analysis.py` expects `deleted_rows.csv`, an intermediate file from the archived `organize_data_2.py` that isn't present either.
- `evaluation/evaluate_model.py` was pointed at `history.pkl` (the file that actually exists at repo root) instead of the `training_history.pkl` name it referenced before this cleanup.
