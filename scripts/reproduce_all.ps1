$ErrorActionPreference = "Stop"

Write-Host "[dmw] Running full JMLR experiment suite..."
python experiments/run_jmlr_full.py
Write-Host "[dmw] Done. Outputs are in experiments/results_jmlr and experiments/figures_jmlr."

