$ErrorActionPreference = "Stop"

# Run evaluation/rag_eval/evals.py using the existing conda env `agent`.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\evaluation\rag_eval\run_evals_agent.ps1

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$evalsPy = Join-Path $PSScriptRoot "evals.py"

Write-Host ("Repo root: {0}" -f $repoRoot)
Write-Host ("Running: conda run -n agent python {0}" -f $evalsPy)

Push-Location $repoRoot
try {
  conda run -n agent python $evalsPy
} finally {
  Pop-Location
}

