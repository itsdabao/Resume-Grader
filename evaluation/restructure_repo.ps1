[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot

function Ensure-Dir([string]$p) {
  if (-not (Test-Path $p)) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
  }
}

try {
  Ensure-Dir "evaluation"
  Ensure-Dir "evaluation\\latex"
  Ensure-Dir "evaluation\\scripts"
  Ensure-Dir "evaluation\\datasets"
  Ensure-Dir "evaluation\\_artifacts"
  Ensure-Dir "evaluation\\_legacy"

  # Move legacy eval docs (optional)
  $docsToMove = @(
    "docs\\3_6_evaluation_tables.tex",
    "docs\\rageval_examples_detailed.tex",
    "docs\\report_evaluation_revised.tex",
    "docs\\report_revised.md",
    "docs\\testset_examples.tex"
  )
  foreach ($p in $docsToMove) {
    if (Test-Path $p) {
      $dest = Join-Path "evaluation\\latex" (Split-Path $p -Leaf)
      if ($PSCmdlet.ShouldProcess("$p -> $dest", "Move")) {
        Move-Item -Force $p $dest
      }
    }
  }

  # Move legacy eval scripts (optional)
  $scriptsToMove = @(
    "scripts\\build_rageval_eval_vi.py",
    "scripts\\export_markdown_nodes_to_jsonl.py"
  )
  foreach ($p in $scriptsToMove) {
    if (Test-Path $p) {
      $dest = Join-Path "evaluation\\scripts" (Split-Path $p -Leaf)
      if ($PSCmdlet.ShouldProcess("$p -> $dest", "Move")) {
        Move-Item -Force $p $dest
      }
    }
  }

  # Move local artifacts (optional)
  $artifactFiles = @(
    "data\\evaluation_results.csv",
    "debug_log.txt",
    "eval_log_full.txt",
    "eval_log_utf8.txt",
    "eval_log_utf8_2.txt"
  )
  foreach ($p in $artifactFiles) {
    if (Test-Path $p) {
      $dest = Join-Path "evaluation\\_artifacts" (Split-Path $p -Leaf)
      if ($PSCmdlet.ShouldProcess("$p -> $dest", "Move")) {
        Move-Item -Force $p $dest
      }
    }
  }

  # Move root helper scripts into a legacy folder (optional).
  # Canonical locations are under `scripts/` and `scripts/dev/`.
  Ensure-Dir "evaluation\\_legacy\\root_scripts"
  $rootScripts = @(
    "cli.py",
    "ingest_global.py",
    "model_test.py",
    "read_presentation.py",
    "reproduce_toxic.py"
  )
  foreach ($p in $rootScripts) {
    if (Test-Path $p) {
      $dest = Join-Path "evaluation\\_legacy\\root_scripts" (Split-Path $p -Leaf)
      if ($PSCmdlet.ShouldProcess("$p -> $dest", "Move")) {
        Move-Item -Force $p $dest
      }
    }
  }

  # Move legacy evaluation tool folders (optional).
  # NOTE: This may be large; review before running.
  foreach ($dir in @("rag_eval", "RAGEval")) {
    if (Test-Path $dir) {
      $dest = Join-Path "evaluation\\_legacy" $dir
      if ($PSCmdlet.ShouldProcess("$dir -> $dest", "Move")) {
        Move-Item -Force $dir $dest
      }
    }
  }

  Write-Host "Done. Review `evaluation/` and then commit changes."
  Write-Host "Tip: run with -WhatIf first to preview."
} finally {
  Pop-Location
}
