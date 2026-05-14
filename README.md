# Distance-Matrix Wasserstein Experiments

This repository contains the reproducibility code for the paper
**Distance-Matrix Wasserstein Statistics for Scalable Gromov--Wasserstein Learning**.

The code reproduces the experiments in the paper:

- approximation--estimation tradeoff for DMW;
- finite-direction behavior of sliced DMW;
- scalability and parameter bottleneck experiments;
- TU graph classification benchmarks;
- structured two-sample testing;
- multi-scale weight ablations.

The repository also includes the CSV tables and PDF figures used in the paper
under `paper_results/`.

## Repository Structure

```text
.
├── experiments/
│   ├── dmw_core.py          # Core DMW, sliced DMW, and metric utilities
│   ├── run_jmlr_full.py     # Main script for the paper experiments
│   └── tu_dmw_full.py       # Earlier self-contained TU benchmark script
├── paper_results/
│   ├── csv/                 # CSV outputs reported in the paper
│   └── figures/             # PDF figures reported in the paper
├── scripts/
│   ├── smoke_test.py        # Quick sanity check without downloading datasets
│   ├── reproduce_all.ps1    # Windows runner for the full suite
│   ├── reproduce_all.sh     # Unix runner for the full suite
│   └── push_to_github.ps1   # Helper for first GitHub upload
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Environment

The scripts require Python 3.10 or newer.  A fresh environment can be created as follows.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The package `POT` provides the `ot` module used for optimal transport and
Gromov--Wasserstein baselines.

## Quick Check

Run a lightweight synthetic smoke test:

```bash
python scripts/smoke_test.py
```

This does not download TU datasets and should finish in seconds.

## Reproducing the Paper Experiments

Run the full JMLR experiment suite:

```bash
python experiments/run_jmlr_full.py
```

On Windows PowerShell:

```powershell
.\scripts\reproduce_all.ps1
```

On Unix-like systems:

```bash
bash scripts/reproduce_all.sh
```

The script writes outputs to:

```text
experiments/results_jmlr/
experiments/figures_jmlr/
experiments/cache_jmlr/
experiments/data/
```

The TU datasets are downloaded automatically from the public TU graph kernel
dataset mirror at `https://www.chrsmrrs.com/graphkerneldatasets/`.  Dataset
archives, extracted raw data, and cache files are intentionally ignored by git.

## Reproducibility Notes

- Graphs are represented as metric measure spaces using normalized shortest-path
  distance and the uniform node measure.
- Sliced DMW samples tuples with replacement and uses Euclidean-normalized
  Gaussian slicing directions in the experiments.
- Classification uses nested cross-validation for hyperparameter selection.
- Exact and entropic GW baselines are evaluated only in regimes where all-pairs
  graph-level GW kernels are computationally meaningful.  Missing entries in
  the paper correspond to intentionally omitted GW baselines rather than failed
  DMW runs.
- Large graph experiments use fixed Monte Carlo budgets for DMW and, where
  needed, node-budgeted metric construction.  This is an implementation
  approximation separate from the population DMW theory.

The committed `paper_results/` directory contains the result files used in the
paper draft.  Re-running the full suite may produce small numerical differences
because of library versions, hardware, and randomized sampling.

## Uploading to GitHub

After creating an empty GitHub repository, run:

```powershell
.\scripts\push_to_github.ps1 -RemoteUrl https://github.com/<USER>/<REPO>.git
```

or manually:

```bash
git remote add origin https://github.com/<USER>/<REPO>.git
git branch -M main
git push -u origin main
```

