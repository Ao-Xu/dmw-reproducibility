#!/usr/bin/env bash
set -euo pipefail

echo "[dmw] Running full JMLR experiment suite..."
python experiments/run_jmlr_full.py
echo "[dmw] Done. Outputs are in experiments/results_jmlr and experiments/figures_jmlr."

