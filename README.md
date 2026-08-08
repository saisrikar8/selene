# SELENE — Space Environment Lunar Exposure Neural Estimator

A deep-learning model that predicts absorbed radiation dose rates on the lunar surface from upstream space-weather data — and reports its own uncertainty. SELENE is a fast, data-driven alternative to heavy physics-based (Monte Carlo / GEANT4) radiation simulators, trained on over 27,000 hours of spacecraft measurements.

## Overview

As missions plan longer stays on the Moon, surface radiation becomes a serious risk: without Earth's atmosphere or magnetic field, the lunar surface is exposed to galactic cosmic rays and solar energetic particles, and the absorbed dose rate swings with space weather. The relationship between upstream particle flux and the resulting dose is nonlinear and noisy — physically simulatable, but expensive to run.

SELENE learns that flux-to-dose relationship directly from data. It ingests particle-flux measurements from the ACE spacecraft and predicts the dose rate recorded by the CRaTER instrument on NASA's Lunar Reconnaissance Orbiter, then quantifies how much each prediction can be trusted.

```

ACE Particle Flux (SIS / CRIS)
|
v
Preprocessing & Daily Re-aggregation
|
v
Correlation-Based Feature Selection
|
v
Neural Network (BatchNorm + LeakyReLU + Dropout)
|
v
Predicted CRaTER Dose Rate
|
v
Monte Carlo Dropout → Uncertainty Estimate

```

## Features

### Data Pipeline
- Downloads and organizes multi-year particle-flux data from the ACE SIS and CRIS detectors
- Aligns ACE flux (hourly) with CRaTER dose (daily) onto a common daily resolution
- Re-aggregation lifted the strongest feature-to-dose correlation from **−0.26 to 0.40**, unlocking the entire result — data alignment mattered more than model architecture

### Feature Selection
- Correlation-based selection to reduce ACE flux channels to the features that actually track dose
- Predicts CRaTER dose rates using only ACE flux inputs

### Neural Network Model
- Fully-connected regression network with batch normalization, LeakyReLU activations, and dropout (0.4 and 0.3 across hidden layers)
- Benchmarked against Linear Regression, XGBoost, and a SAINT transformer baseline

### Uncertainty Quantification
- Monte Carlo Dropout: dropout stays active at inference, and **100 stochastic forward passes** per input yield a distribution rather than a point estimate
- Predictive variance correlates with actual error (Pearson **r = 0.46, p < 10⁻¹⁴⁴**) — the uncertainty is meaningful, not decorative

## Architecture

```
      +----------------------+
      |   ACE Flux (SIS/CRIS)|
      +----------------------+
                 |
                 v
      +----------------------+
      |  Daily Re-aggregation|
      +----------------------+
                 |
                 v
      +----------------------+
      | Feature Selection    |
      +----------------------+
                 |
                 v
      +----------------------+
      |   Neural Network     |
      +----------------------+
          /              \
         v                v
+----------------+  +------------------+
| Dose Estimate  |  | MC Dropout       |
+----------------+  | Uncertainty      |
                    +------------------+
```

## Results

Benchmarked on held-out data against several baselines:

| Model | MAE | R² | MAPE |
|---|---|---|---|
| **SELENE (ANN)** | **0.00429** | **0.834** | **14.26%** |
| SELENE + MC Dropout | 0.00442 | 0.827 | 14.93% |
| XGBoost | 0.00816 | 0.011 | 42.83% |
| Linear Regression | 0.00878 | 0.388 | 60.67% |
| SAINT (transformer) | 0.885 | −2135 | 79.27% |

SELENE captured the nonlinear flux-to-dose relationship (R² = 0.834), while XGBoost effectively failed (it output nearly the same value for every input), linear regression underfit, and the SAINT transformer broke down entirely — more capacity was not automatically better on noisy sensor data. Using MC Dropout variance to flag likely-wrong predictions, a 90th-percentile threshold gave the best precision/recall/F1 trade-off (**51.7%** each), with the highest-variance cases landing exactly where the errors were largest.

## Technologies

* **Language:** Python
* **Concepts:**

  * Deep Learning (Keras / TensorFlow)
  * Regression on Physical Time-Series Data
  * Monte Carlo Dropout & Uncertainty Quantification
  * Correlation-Based Feature Selection
  * Gradient Boosting (XGBoost) and Transformer (SAINT) Baselines
  * Space Weather / Radiation Physics

## Project Structure

```
pipeline/         # Data download, alignment, and daily aggregation
  download_sis_data.py
  download_epam_data_async.py
  avg_data.py
  organize_data.py
models/           # Model training and baselines
  train_model.py        # SELENE neural network
  montecarlotrain.py    # MC Dropout uncertainty variant
  train_xgboost.py      # XGBoost baseline
  train_SAINT.py        # SAINT transformer baseline
  linregtest.py         # Linear regression baseline
evaluation/       # Metrics, error and data analysis
  evaluate_model.py
  error_analysis.py
  data_analysis.py
book_chapter/     # Interpretability & calibration experiments for the publication
```

## Future Improvements

Potential extensions:

* Incorporate additional upstream detectors (GOES, EPAM) for cross-instrument robustness
* Calibrated uncertainty (temperature scaling / conformal prediction) beyond MC Dropout
* Sequence models to capture temporal lag between flux events and dose response
* Real-time inference against live ACE data feeds

## Learning Outcomes

This project taught me that in applied ML on physical data, careful preprocessing often outweighs model choice — the single largest gain came from aligning mismatched sampling rates, not from a fancier architecture. I also learned to treat uncertainty as a first-class output: for a safety-relevant prediction, "here's a number and how much to trust it" is far more useful than the number alone.

## Publication

SELENE was published in the proceedings of the World Congress in Computer Science, Computer Engineering & Applied Computing (CSCE) by Springer Nature. Publication listing: [Google Scholar](https://scholar.google.com/citations?user=31gyvjkAAAAJ&hl=en).

---

**Author:** Sai Srikar Tummala
