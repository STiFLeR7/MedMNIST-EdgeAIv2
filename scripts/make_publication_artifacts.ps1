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

# Datasets & model tags
$Datasets = @("ham10000","medmnist","isic")
$Students = @("resnet18","mbv2","effb0")
$Teacher = "resnet50"

# 0) Ensure dirs
New-Item -ItemType Directory -Force -Path $OutFigs,$OutTables | Out-Null

# 0.1) Ensure predictions exist (creates test_predictions.npz if missing)
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

# 1) Per-class metrics, confusion matrices, PR/F1 tables
foreach ($ds in $Datasets) {
  & $PY ".\tools\perclass_metrics.py" `
    --dataset $ds `
    --reports-root ".\reports" `
    --models "$Teacher,$($Students -join ',')" `
    --out-figs "$OutFigs" `
    --out-tables "$OutTables" `
    --save-latex
}

# 2) Ablations (α, τ, β) aggregation + plots
$rebuildFlag = @()
if ($RebuildAblations) { $rebuildFlag = @("--rebuild") }  # <-- correct boolean switch

& $PY ".\tools\ablate_kd.py" `
  --datasets "$($Datasets -join ',')" `
  --reports-root ".\reports" `
  --out-figs "$OutFigs" `
  --out-tables "$OutTables" `
  @rebuildFlag

# 3) Efficiency table (Params/FLOPs/Latency/RAM)
& $PY ".\tools\efficiency_profile.py" `
  --dataset-list "$($Datasets -join ',')" `
  --models "$Teacher,$($Students -join ',')" `
  --imgsz 224 `
  --device $DeviceGPU `
  --batches "1,2,4,8" `
  --out-tables "$OutTables" `
  --out-figs "$OutFigs"

# 4) Clinical interpretability (Grad-CAM)
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

# 5) Optional: t-SNE teacher vs students
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
Write-Host "✅ Publication artifacts generated under $OutFigs and $OutTables."
