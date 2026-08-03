# Design Spec: Lunar Radiation Book Chapter (SELENE Extension)

**Date:** 2026-07-26
**Status:** Approved (design), pending user spec review
**Project:** lunar-radiation
**Deliverable:** Springer book-chapter extension of the published SELENE conference paper

---

## 1. Purpose & Publisher Constraints

Extend the published conference paper into a Springer Nature **book chapter**. The
publisher's requirements that this design must satisfy:

1. **≥50% new material** vs. the published paper; final length **15–25 pages**.
2. **Cite the original paper** explicitly (the chapter builds on it).
3. **Different title and abstract** from the published paper.
4. **Same authors** as published (order may change; new authors may be added).
   → Decision: **same three, same order** — Tummala, Bhatia, Chun.
5. **Figures must be grayscale-safe** (book may print in B/W; no color-only encoding).
6. High-resolution figure files exported **separately** (not only embedded).
7. No orphan citations (every reference cited in text; every in-text cite has an entry).
8. Format: **LaTeX**, Springer book-chapter template (svmono/llncs family), single column.

> Note: the pasted publisher instructions included a contact email. It is treated as
> reference only — no email will be sent as part of this work.

## 2. Source of Truth (Published Paper)

- **File:** `Lunar_Research/main.tex` (+ `Lunar_Research/references.bib`)
- **Title:** *SELENE: A Deep Learning Model for Predicting Lunar Absorbed Radiation Dose Rates*
- **Authors:** Sai Srikar Tummala\*, Mittansh Bhatia\*, Robert Chun (SJSU)
- **Core:** SELENE, a feedforward ANN predicting CRaTER daily absorbed dose rates from
  ACE SIS/CRIS particle-flux features (~27,000 hourly records aggregated to daily; 18
  features after Pearson feature selection).
- **Headline results:** SELENE MAPE 14.26%, R² 0.8342; beats Linear Regression, XGBoost,
  and SAINT (SAINT collapsed: R² −2135). Includes MC-Dropout uncertainty + error analysis.

The chapter must **cite** this paper (add a BibTeX entry once venue/year are confirmed by
the author) and clearly attribute all recapped material to it.

## 3. Framing / Thesis (Approach A — approved)

The paper proved SELENE is **fast and accurate**. The chapter's new thesis: to be usable
in mission operations a model must also be **trustworthy** — **calibrated in its
uncertainty** and **physically interpretable**. This unifies the two new experiments under
one operational-deployment story, clearly distinct from the original contribution.

- **Working title:** *Toward Trustworthy Lunar Radiation Nowcasting: Calibrated Uncertainty
  and Physical Interpretability in Data-Driven Dose-Rate Estimation*
- **Abstract direction:** recap that fast ML dose estimation is feasible (cite SELENE
  paper), then frame the two new contributions (calibration analysis; physics-grounded
  interpretability/ablation) as prerequisites for operational deployment in Artemis-era
  missions.

## 4. Chapter Outline (~20 pages)

Tags: **[NEW]** net-new, **[EXPANDED]** materially grown, **[RECAP]** condensed from paper.

1. **Introduction** — [EXPANDED] Artemis motivation; trustworthiness (not just speed) gates
   operational use; state two new contributions; cite SELENE paper. (~1.5 pg)
2. **Background: The Lunar Radiation Environment** — [NEW] GCR vs SEP origins; absorbed
   dose vs LET vs dose equivalent & quality factors; unshielded Moon; CRaTER/ACE
   instrumentation. (~3 pg)
3. **Related Work** — [EXPANDED] Physics models (REDMoon/GEANT4, HZETRN, OLTARIS);
   dosimetry (LND/Chang'e-4); ML forecasting; **new subsection on UQ & interpretability in
   scientific ML**. (~2.5 pg)
4. **The SELENE Model (Recap)** — [RECAP] Condensed data pipeline, ANN architecture,
   headline results — attributed to the published paper. (~2 pg)
5. **Physical Interpretability & Ablation** — [NEW, Experiment 1] (~3.5 pg)
6. **Calibrated Uncertainty for Operational Risk** — [NEW, Experiment 2] (~3.5 pg)
7. **Discussion** — [NEW] Interpretability + calibration together for mission deployment;
   failure modes; limitations. (~1.5 pg)
8. **Future Work & Conclusion** — [EXPANDED] Temporal models, Mars extension, onboard
   deployment. (~1 pg)

**Balance:** new/expanded ≈ 80%, recap ≈ 20% → clears ≥50%-new requirement.

## 5. Experiment Specifications

Both experiments run locally on **CPU** (the ANN is small: 18 features, ~27k samples).

### Experiment 1 — Physical Interpretability & Ablation
- Reload/retrain the SELENE ANN on `aligned_averaged_by_crater.csv` features.
- **Permutation importance** (sklearn `permutation_importance`) over the 18 features.
- **SHAP** values (KernelExplainer or DeepExplainer) for per-feature attribution.
- **Ablation:** retrain with top-k features (k = 2, 4, 8, 12, 18); plot metric vs k.
- **Cross-check:** XGBoost gain-based feature importance vs the ANN attributions.
- **Physics interpretation:** map dominant features to ion species / energy bands.
- Outputs: grayscale-safe importance bar chart, ablation curve, SHAP summary (B/W-safe).

### Experiment 2 — Calibrated Uncertainty
- Methods compared: **MC-Dropout** (existing), **deep ensemble** (N≈10 ANNs),
  **quantile regression** (pinball-loss head).
- Metrics: **ECE** (expected calibration error via binning), **reliability diagram**,
  **PICP@90%** (prediction-interval coverage), **MPIW** (mean interval width), sharpness.
- Deliverable: which method gives best calibration/sharpness trade-off for operational
  risk-flagging; calibrated thresholds for flagging low-confidence predictions.
- Outputs: reliability diagram, coverage/width table, calibration-vs-sharpness figure.

## 6. Build Plan

- **Step 0 — Environment:** create a dedicated `.venv` (or conda env) and install
  numpy, pandas, scikit-learn, scipy, xgboost, tensorflow (Apple-Silicon build), shap,
  matplotlib. One-time ~15–20 min / few-hundred-MB download. Pin versions in
  `requirements.txt` for reproducibility.
- **Figures:** 300 dpi, grayscale-safe (patterns/markers/linestyles, not color alone);
  export each as a separate high-res file **and** embed in the chapter.
- **LaTeX project:** new dir `Lunar_Research_Book_Chapter/` with Springer template,
  `main.tex`, `references.bib` (extend the existing bib; add citation to the SELENE paper).
- **Reference hygiene:** audit for orphan citations before finalizing.

## 7. Open Items (need author input during implementation)

- **Citation to original paper:** exact venue, year, DOI/pages for the SELENE conference
  publication (needed for the required self-citation BibTeX entry).
- **Author emails:** keep as published (`tummalasaisrikar@gmail.com`, etc.) unless the
  author prefers professional addresses (publisher preference, not required).
- **Git:** `lunar-radiation` is tracked inside the shared `surentech-org/surentech-server`
  repo on `main`. Do not auto-commit to shared `main`; branch or defer commits to the author.

## 8. Success Criteria

- Chapter compiles in the Springer LaTeX template, 15–25 pages, single column.
- ≥50% new/expanded material; different title & abstract; cites the SELENE paper.
- Same three authors, same order.
- Both experiments produce real numbers + grayscale-safe figures exported at 300 dpi.
- No orphan citations in either direction.
