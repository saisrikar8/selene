# Book Chapter Runbook

Everything is authored. This environment (Claude's sandbox) cannot read the
Desktop data or run Python/LaTeX against it, so **these commands must run in your
own Terminal**, which has file access. Run them from the repo root
(`lunar-radiation/`).

## 1. Set up the environment (once)
```bash
cd ~/Desktop/work/research/lunar-radiation
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r book_chapter/requirements.txt
```

## 2. Sanity-check the reproduced SELENE model
```bash
cd book_chapter
../.venv/bin/python -m pytest test_selene_common.py -v
../.venv/bin/python -c "import selene_common as sc; X,y,_=sc.load_data(); \
Xtr,Xv,ytr,yv=sc.split(X,y); m,_=sc.train_selene(Xtr,ytr,Xv,yv); \
print(sc.metrics(yv, m.predict(Xv,verbose=0)))"
```
Record the printed `r2` and `mape` — these fill the two `\FILL{selene ...}`
slots in the SELENE recap section.

## 3. Run the two experiments (produce real numbers + figures)
```bash
cd book_chapter
../.venv/bin/python exp1_interpretability.py   # writes outputs/exp1_metrics.json + figs
../.venv/bin/python exp2_calibration.py        # writes outputs/exp2_metrics.json + figs
```
(The deep ensemble trains 10 models; allow a few minutes on CPU.)

## 4. Move figures into the chapter
```bash
mkdir -p ../Lunar_Research_Book_Chapter/figures
cp outputs/*.pdf ../Lunar_Research_Book_Chapter/figures/
```

## 5. Fill the `\FILL{...}` placeholders
Open `Lunar_Research_Book_Chapter/main.tex`, search for `\FILL`, and replace each
with the real value from:
- `outputs/exp1_metrics.json` — `base_metrics`, `permutation_importance`
  (top features), `ablation` (R^2 vs k), `xgb_importance`.
- `outputs/exp2_metrics.json` — `results.{mc_dropout,deep_ensemble,quantile}`
  → `picp`, `mpiw`, `ece`, `rmse` for the table; pick the lowest-`ece` method as
  "best method".
Paste back the two JSON files here and I'll fill them for you if you prefer.

## 6. Verify no orphan citations / no leftover placeholders
```bash
cd book_chapter
python check_finalization.py    # exit 0 when clean
```

## 7. Compile the chapter
```bash
cd ../Lunar_Research_Book_Chapter
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdfinfo main.pdf | grep Pages   # expect 15-25
```
> Class note: `main.tex` uses Springer `llncs`. If your editor sent the `svmono`
> book template instead, change `\documentclass` and the author/`\institute`
> block to match svmono; the body is unchanged. Compiling on Overleaf with the
> Springer template is the easiest path.

## Publisher checklist (all satisfied by design)
- [x] Different title & abstract from the SELENE paper
- [x] Cites the original paper (`tummala2025selene`) — **confirm its venue/year in
      `references.bib`**
- [x] Same authors, same order (Tummala, Bhatia, Chun)
- [x] Grayscale-safe figures at 300 dpi, exported as separate files
- [x] ≥50% new material (2 new experiments + new background/related-work)
- [ ] Fill every `\FILL{}` slot with real numbers (step 5) before submission
