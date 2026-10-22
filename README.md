# quantum-kernel-inference

Code, run records and figures for the manuscript

> *When Does Adaptive Inference Measurement Allocation Pay Off in Quantum Kernel Classification? An Overhead-Aware Performance Study*

**Everything here is a classical simulation.** Kernel entries are computed exactly
for 4-qubit feature maps (NumPy state vectors). Finite-shot estimates are
**binomial draws with the exact kernel entry as success probability**. There is
no quantum hardware, no readout error, no drift, no compilation. The results
say nothing about hardware performance.

## Layout

```
scripts/main_study.py          main study (v6.1): 3 workloads x 30 splits, B in {640,1280,2560}
scripts/robustness_study.py    one-factor-at-a-time robustness grid (11 configurations x 3 workloads, 12 splits, B=1280)
scripts/make_figures.py        regenerates the three PDF figures from the result CSVs
results/                       v6_1_results.csv, v6_1_robustness_summary.csv, v6_1_robustness_rows.csv
audits/                       v6_1_dataset_audit.json, v6_1_robustness_manifest.json
logs/                         console output of the two runs
figures/                      fig_variance.pdf, fig_primary.pdf, fig_robust.pdf
```

`robustness_study.py` is `main_study.py` plus (i) an `SVC_C` / `KBIAS` setting
applied in `run_split`, (ii) a `SKIPPED` check, and (iii) the robustness section
and `main_robust()` at the end of the file. It is self-contained; neither script
imports the other. `results/`, `audits/` and `logs/` hold the outputs of a run;
see the README in each folder.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reported runs: Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1
(Linux x86_64). All randomness is seeded (split seeds 0..29; per-split, per-draw,
per-budget, per-replicate generator seeds), so with the same library versions the
printed statistics should reproduce. I have not verified bit-for-bit
reproduction across other NumPy/scikit-learn versions or platforms. Timing
columns (`alloc_s`, `pilot_s`, ...) are machine-dependent and will not match.

## Datasets

The raw datasets are **not** included. The scripts read them from fixed paths
under `/kaggle/input/` (the layout of a Kaggle notebook). Files must be the
**Kaggle copies below**, because the recorded SHA-256 hashes refer to those files
(the UCI originals have different layouts and will not match).

| Workload | Kaggle dataset (`kaggle datasets download -d ...`) | Path the script expects | Rows | Pos / Neg | Expected SHA-256 |
|---|---|---|---|---|---|
| AI4I 2020 | `stephanmatzka/predictive-maintenance-dataset-ai4i-2020` | `/kaggle/input/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/ai4i2020.csv` | 10,000 | 339 / 9,661 | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| Breast Cancer Wisconsin (Diagnostic) | `uciml/breast-cancer-wisconsin-data` | `/kaggle/input/datasets/uciml/breast-cancer-wisconsin-data/data.csv` | 569 | 212 / 357 | `1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33` |
| Pima Indians diabetes | `kumargh/pimaindiansdiabetescsv` | any `/kaggle/input/**/*pima*.csv`; used: `/kaggle/input/datasets/kumargh/pimaindiansdiabetescsv/pima-indians-diabetes.csv` | 768 | 268 / 500 | `c1e0f7d59d2e366cff1ab9fee1796e1e381ea5da64c6d96ef4baabf1eda5f238` |

Notes:

* **On Kaggle:** attach the three datasets via *Add Input*; the paths above then exist.
* **Locally:** the paths are hard-coded, so recreate them, e.g.
  ```bash
  sudo mkdir -p /kaggle/input/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020 \
                /kaggle/input/datasets/uciml/breast-cancer-wisconsin-data \
                /kaggle/input/datasets/kumargh/pimaindiansdiabetescsv
  # then copy ai4i2020.csv, data.csv, pima-indians-diabetes.csv into the matching folders
  ```
* **Verify hashes** before running: `sha256sum ai4i2020.csv data.csv pima-indians-diabetes.csv`.
  Each run also records them (`audits/*.json`, and the `=== RUN AUDIT ===` block of the logs) and
  prints a warning if row or class counts differ from the documented sizes.
* **Breast Cancer fallback:** if `data.csv` is absent, the script silently uses scikit-learn's bundled
  copy of the same WDBC data (audit source = "scikit-learn bundled copy"). Hash and row order differ,
  so splits and numbers will not match the reported ones. Use the Kaggle file for reproduction.
* **Pima:** the file has no header; the classic UCI column order is *assumed* (row and class counts
  match, which supports but does not prove it). Zeros are kept as recorded (no imputation).
* **AI4I fail-closed:** if the AI4I CSV is missing, the scripts exit unless `ALLOW_STANDIN=1` is set,
  which substitutes generated data **for code testing only (never a result)**. Pima is skipped if absent.

### Redistribution terms

I did not copy any raw data into this repository. As I understand them (verify on the source pages before
relying on this), AI4I 2020 and Breast Cancer Wisconsin (Diagnostic) are UCI Machine Learning Repository
datasets released under CC BY 4.0 (DOIs 10.24432/C5HS5C and 10.24432/C5DW2B), which would permit
redistribution with attribution; the Kaggle copies may carry their own terms. The licensing of the
Pima file I could not establish (it is a third-party Kaggle copy of a dataset of NIDDK origin), so it
should not be bundled without checking. Providing acquisition instructions plus hashes, as above, avoids
the question.

## Running

Both scripts write their outputs to the **current directory**, so run them from `results/`:

```bash
cd results

# Full runs (times on the Kaggle CPU notebook used for the paper: about 17 min and 9.5 min)
python ../scripts/main_study.py       | tee ../logs/main_study.txt
python ../scripts/robustness_study.py | tee ../logs/robustness_study.txt

mv v6_1_dataset_audit.json v6_1_robustness_manifest.json ../audits/
cd ..
python scripts/make_figures.py --results results --outdir figures
```

Small tests (use a scratch directory so the real outputs are not overwritten; 2 splits per workload):

```bash
mkdir -p /tmp/qk_test && cd /tmp/qk_test
python -c "import sys; sys.path.insert(0,'<repo>/scripts'); import main_study as m; m.main(nsplits=2)"
python -c "import sys; sys.path.insert(0,'<repo>/scripts'); import robustness_study as m; m.main_robust(nsplits=2)"
python <repo>/scripts/make_figures.py --results . --outdir ./figs
# no datasets attached? add:  ALLOW_STANDIN=1   (code test only; the numbers mean nothing)
```

Both scripts first run a start-up self-test of the integer allocator (heap-greedy reference,
brute force, a known counterexample, all-zero rows, budget exactness, the pilot-1/control-1
identity) and abort on failure. The packaging smoke test (2 splits, stand-in AI4I, scikit-learn
Breast Cancer data, Python 3.12, NumPy 2.4, scikit-learn 1.8) ran end to end, and `make_figures.py`
reproduced the intervals printed by `main_study.py`. That test is not a reproduction of the reported numbers.

## Coupled versus independent random streams

| Experiment | Streams | Measurement replicates | Policies |
|---|---|---|---|
| `main_study.py`, main analysis (all tables except the sensitivity columns) | **coupled** (all policies share generator seeds) | 3 | all, incl. `oracle_pilot_sp*` |
| `main_study.py`, sensitivity (`SENSITIVITY` blocks; independent columns of the primary table and figure; known-null check) | **independent** (own stream per policy, seeded from `crc32(policy)`) | 4 | no `oracle_pilot_sp*` |
| `robustness_study.py` | **coupled** only (`RUN_SENSITIVITY=False`) | 3 | `uniform, coef, oracle_wf, pilot_sp{2,4}, ctrl_sp{2,4}` |

Both schemes are in the same file, `v6_1_results.csv` (column `scheme`). The ratio of interval widths between the
schemes is not a controlled estimate of the coupling effect, because the replicate counts differ.

## Where each manuscript table and figure comes from

Labels are the LaTeX `\label`s (numbering follows compile order). "Log" means the per-workload
blocks of `logs/main_study.txt` unless stated; all of them can be recomputed from `results/v6_1_results.csv`.

| Manuscript item | Source |
|---|---|
| `tab:related` | no data (literature table) |
| `tab:workloads` | Rows / positives: audit JSON. Test size: `N_TEST_BY`. $m$, AP, AUC, balanced accuracy: log, *Trained-model diagnostics* (`m_sv`, `average_precision`, `auc`, `bal_acc`) |
| `tab:variance` | Log, *Variance headroom*: `coef`, `oracle_wf` at B = 640/1280/2560; `m_sv`, `cv_beta` from *Trained-model diagnostics*. $H = 1 - V_\text{oracle}/V_\text{coef}$ and "captured by coef" $= (1-V_\text{coef})/(1-V_\text{oracle})$ are computed by hand from those ratios (no script prints them) |
| `fig:variance` → `figures/fig_variance.pdf` | `make_figures.py`; coupled, B = 1280, ratio of mean conditional variances for `coef`, `oracle_wf`, `pilot_sp1/2/4`, `oracle_pilot_sp2` |
| `tab:pilots` (top block) | Log, *Variance headroom* (`coef`, `ctrl_sp`, `pilot_sp`, `oracle_pilot_sp` at B = 1280) and *Recovery* |
| `tab:pilots` (bottom block) | Log, *Exploratory paired differences*, B = 1280: AP of `pilot_spK - ctrl_spK` (value of information) and `ctrl_spK - coef` (cost of acquisition) |
| `tab:breakeven` and the $Km/H$ thresholds in the text | Log, *Variance headroom*: `pilot_frac` and `oracle_pilot_spK` vs `coef` for all budgets. The Y/N marks and $Km/H$ are derived by hand |
| `tab:primary`, `fig:primary` → `figures/fig_primary.pdf` | Coupled column: log, *PRIMARY* (coupled). Independent column: log, *SENSITIVITY*, `coef - uniform` at B = 1280. `make_figures.py` prints the same six intervals |
| `tab:ap` | Log, *Predictive metrics*, `average_precision`, B = 1280 |
| Known-null comparison (text, Sec. "Predictive quality") | Log, *SENSITIVITY*, rows `pilot_sp1 - ctrl_sp1`, independent columns |
| `tab:calib` | Log, *Predictive metrics*, `emp_rmse` / `theory_rmse`, B = 1280 |
| Classical-cost paragraph | AI4I block only, *Classical simulation cost per query batch* (machine-dependent) |
| `tab:robust`, `fig:robust` → `figures/fig_robust.pdf`; robustness bullets | `logs/robustness_study.txt` and `results/v6_1_robustness_summary.csv` (columns `V_coef`, `V_oracle`, `dAP coef-uni`, `lo/hi coef-uni`, `dAP sp2-coef`, `dAP sp2-ctrl`, ...); the 33/33, 29/33, ... counts are the final *Coverage across the grid* line. The 3–11% headroom range is derived by hand from `V_coef` and `V_oracle` |
| `tab:audit` and Appendix A text | `audits/v6_1_dataset_audit.json` and the *RUN AUDIT* block |
| `tab:budget` | Log, *Exploratory paired differences* (coupled) and *SENSITIVITY* (independent), `coef - uniform` at all three budgets |

Quantities derived by hand (marked above) are simple arithmetic on printed ratios; this repository does not
contain a script that produces them.

## Citation

See `CITATION.cff` (repository URL / DOI still to be added). License: MIT for the code (`LICENSE`); datasets are
not covered.
