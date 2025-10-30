param(
  [string]$RepoRoot = ".",
  [string]$OutFigs = ".\figs",
  [string]$OutTables = ".\tables",
  [string]$DeviceGPU = "cuda",    # or "cpu"
  [int]$TSNESamples = 1500,
  [switch]$RebuildAblations
)

$ErrorActionPreference = "Stop"
$PY = "python"

# -------------------------
# Datasets & model tags
# -------------------------
$Datasets = @("ham10000","medmnist","isic")
$Students = @("resnet18","mbv2","effb0")   # keep CLI tags consistent with tools
$Teacher  = "resnet50"

# -------------------------
# Utilities
# -------------------------
function Ensure-Dir([string]$p) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Copy-WithPrefix {
  param(
    [string]$SourceDir,
    [string]$Pattern,
    [string]$DestDir,
    [string]$Prefix
  )
  if (-not (Test-Path $SourceDir)) { return }
  Ensure-Dir $DestDir
  Get-ChildItem -Path $SourceDir -Filter $Pattern -File -Recurse | ForEach-Object {
    $dest = Join-Path $DestDir ($Prefix + $_.Name)
    Copy-Item $_.FullName $dest -Force
  }
}

# -------------------------
# 0) Ensure output dirs
# -------------------------
Ensure-Dir $OutFigs
Ensure-Dir $OutTables

# -------------------------
# 0.1) Export predictions if missing
# -------------------------
foreach ($ds in $Datasets) {
  foreach ($m in @($Teacher) + $Students) {
    & $PY ".\tools\export_predictions.py" `
      --dataset $ds `
      --model $m `
      --out-root ".\reports" `
      --device $DeviceGPU `
      --force-missing-only
  }
}

# -------------------------
# 0.2) Backfill ablation records from models/ (so ablate_kd has material)
# -------------------------
if (Test-Path ".\tools\backfill_ablation_from_modelsdir.py") {
  & $PY ".\tools\backfill_ablation_from_modelsdir.py"
}

# -------------------------
# 1) Per-class metrics (tables + confusion + PR)
# -------------------------
foreach ($ds in $Datasets) {
  & $PY ".\tools\perclass_metrics.py" `
    --dataset $ds `
    --reports-root ".\reports" `
    --models "$Teacher,$($Students -join ',')" `
    --out-figs "$OutFigs" `
    --out-tables "$OutTables" `
    --save-latex
}

# -------------------------
# 2) Ablations (α, τ, β) aggregation + plots
# -------------------------
$rebuildFlag = @()
if ($RebuildAblations) { $rebuildFlag = @("--rebuild") }

& $PY ".\tools\ablate_kd.py" `
  --datasets "$($Datasets -join ',')" `
  --reports-root ".\reports" `
  --out-figs "$OutFigs" `
  --out-tables "$OutTables" `
  @rebuildFlag

# -------------------------
# 3) Efficiency profiling (Params / FLOPs / Latency / RAM)
#     Uses the CLI tool you wired to THOP-safe logic.
# -------------------------
& $PY ".\tools\efficiency_profile.py" `
  --dataset-list "$($Datasets -join ',')" `
  --models "$Teacher,$($Students -join ',')" `
  --imgsz 224 `
  --device $DeviceGPU `
  --batches "1,2,4,8" `
  --out-tables "$OutTables" `
  --out-figs "$OutFigs"

# -------------------------
# 4) Clinical interpretability (Grad-CAM)
#    NB: tools/gradcam_viz.py must be THOP-safe (we updated logic in the notebook).
# -------------------------
foreach ($ds in $Datasets) {
  & $PY ".\tools\gradcam_viz.py" `
     --dataset $ds `
     --reports-root ".\reports" `
     --models "$($Students -join ',')" `
     --teacher "$Teacher" `
     --samples-per-class 25 `
     --device $DeviceGPU `
     --out-figs "$OutFigs"
}

# -------------------------
# 5) t-SNE (teacher vs students), Procrustes-aligned
#    NB: tools/tsne_features.py should deep-copy + strip THOP as in notebook.
# -------------------------
foreach ($ds in $Datasets) {
  & $PY ".\tools\tsne_features.py" `
    --dataset $ds `
    --reports-root ".\reports" `
    --teacher "$Teacher" `
    --students "$($Students -join ',')" `
    --samples $TSNESamples `
    --device $DeviceGPU `
    --out-figs "$OutFigs" `
    --out-tables "$OutTables"
}

# -------------------------
# 6) Consolidate artifacts with dataset-prefixed filenames
#    (Avoid name collisions across datasets)
# -------------------------
# Figures to consolidate
$FigPatterns = @(
  "*_confmat.png", "*_pr_curves.png", "*gradcam_panel.png",
  "tsne_teacher_vs_*.png", "ablation_*.*", "efficiency_*.*"
)

# Tables to consolidate
$TablePatterns = @(
  "*perclass_metrics.csv", "*perclass_metrics.tex",
  "efficiency_*.csv", "efficiency_*.tex",
  "ablation_*.*"
)

# Per-dataset subdirs (in case tools saved under dataset-nested dirs)
$PerDsFigDir = @{
  "ham10000" = ".\figs\ham10000"
  "medmnist" = ".\figs\medmnist"
  "isic"     = ".\figs\isic"
}
$PerDsTabDir = @{
  "ham10000" = ".\tables\ham10000"
  "medmnist" = ".\tables\medmnist"
  "isic"     = ".\tables\isic"
}

foreach ($ds in $Datasets) {
  $prefix = "$ds`_"
  # Copy figures from per-dataset subdir (if exists)
  if ($PerDsFigDir.ContainsKey($ds)) {
    foreach ($pat in $FigPatterns) { Copy-WithPrefix -SourceDir $PerDsFigDir[$ds] -Pattern $pat -DestDir $OutFigs -Prefix $prefix }
  }
  # Copy figures from root figs (if some tools wrote directly under $OutFigs)
  foreach ($pat in $FigPatterns) { Copy-WithPrefix -SourceDir $OutFigs -Pattern $pat -DestDir $OutFigs -Prefix $prefix }

  # Copy tables from per-dataset subdir (if exists)
  if ($PerDsTabDir.ContainsKey($ds)) {
    foreach ($pat in $TablePatterns) { Copy-WithPrefix -SourceDir $PerDsTabDir[$ds] -Pattern $pat -DestDir $OutTables -Prefix $prefix }
  }
  # Copy tables from root tables (if some tools wrote directly under $OutTables)
  foreach ($pat in $TablePatterns) { Copy-WithPrefix -SourceDir $OutTables -Pattern $pat -DestDir $OutTables -Prefix $prefix }
}

Write-Host "✅ Publication artifacts generated and consolidated under $OutFigs and $OutTables."
