<#
Phase 3 Runner — ISIC (HAM-mode classification) : Teacher + Students
- Calls external_src/eval/eval_pack_isic.py with --mode ham
- Requires a HAM label_map.json; optional ISIC→HAM mapping to collapse classes if needed.

Usage (from repo root):
  pwsh -File .\scripts\phase3_isic.ps1 `
    -ISICRoot ".\data\ISIC" `
    -TeacherCkpt ".\models\teachers\isic_resnet50_v2\ckpt-best.pth" `
    -LabelMap ".\models\teachers\runs_ham10000_resnet50\label_map.json" `
    -IsicToHamMap ".\external_data\mappings\isic_to_ham.json" `
    -Students "efficientnet_b0|.\models\students\isic_kdat_all\student_effb0_isic_kdat|student_effb0_isic",
              "mobilenet_v2|.\models\students\isic_kdat_all\student_mbv2_isic_kdat|student_mbv2_isic",
              "resnet18|.\models\students\isic_kdat_all\student_res18_isic_kdat|student_resnet18_isic" `
    -OutRoot ".\reports\phase3_isic" `
    -Device cuda `
    -DoRobustness:$true `
    -NumWorkers 0
#>

[CmdletBinding()]
param(
  [string]$ISICRoot = ".\data\ISIC",

  # You may pass either a directory (will pick ckpt-best.pth or ckpt-last.pth) or a direct .pth file.
  [string]$TeacherCkpt = ".\models\teachers\isic_resnet50_v2\ckpt-best.pth",

  # Students entries: "arch|ckpt_or_dir|tag"
  [string[]]$Students = @(
    "efficientnet_b0|.\models\students\isic_kdat_all\student_effb0_isic_kdat|student_effb0_isic",
    "mobilenet_v2|.\models\students\isic_kdat_all\student_mbv2_isic_kdat|student_mbv2_isic",
    "resnet18|.\models\students\isic_kdat_all\student_res18_isic_kdat|student_resnet18_isic"
  ),

  [string]$OutRoot = ".\reports\phase3_isic",

  # REQUIRED for --mode ham
  [Parameter(Mandatory=$true)] [string]$LabelMap,
  # Optional mapping file (if class names don’t exactly match)
  [string]$IsicToHamMap = "",

  # Eval/runtime knobs
  [ValidateSet("cuda","cpu")] [string]$Device = "cuda",
  [int]$EvalBatchSize = 32,
  [int]$ImgSize = 224,
  [int]$Bins = 15,
  [int]$NumWorkers = 4,

  # Robustness on ISIC test
  [switch]$DoRobustness = $true,
  [string]$CorruptionLevels = "gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6",

  # Efficiency sweeps
  [int]$LatGPURepeats = 100,
  [int]$LatGPUWarmup  = 20,
  [string]$LatGPUBatches = "1,2,4,8",
  [int]$LatCPURepeats = 200,
  [int]$LatCPUWarmup  = 30,
  [string]$LatCPUBatches = "1,2,4"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
  if (-not $p) { throw "Ensure-Dir received null/empty path." }
  if (-not (Test-Path -LiteralPath $p)) {
    New-Item -ItemType Directory -Path $p | Out-Null
  }
  return (Resolve-Path -LiteralPath $p).Path
}
function Resolve-Existing([string]$p, [switch]$AllowEmpty = $false) {
  if ($AllowEmpty -and [string]::IsNullOrWhiteSpace($p)) { return "" }
  if (-not (Test-Path -LiteralPath $p)) { throw "Path not found: $p" }
  return (Resolve-Path -LiteralPath $p).Path
}
function Resolve-Ckpt([string]$p) {
  # Accept either: directory (auto-pick best/last) or direct file
  if (-not (Test-Path -LiteralPath $p)) {
    throw "Checkpoint not found: $p"
  }
  $item = Get-Item -LiteralPath $p
  if ($item.PSIsContainer) {
    $best = Join-Path $item.FullName "ckpt-best.pth"
    $last = Join-Path $item.FullName "ckpt-last.pth"
    if (Test-Path -LiteralPath $best) { return (Resolve-Path $best).Path }
    if (Test-Path -LiteralPath $last) { return (Resolve-Path $last).Path }
    throw "No ckpt-best.pth or ckpt-last.pth in $($item.FullName)"
  } else {
    return (Resolve-Path -LiteralPath $p).Path
  }
}
function Py() {
  param([Parameter(ValueFromRemainingArguments=$true)] [string[]]$Args)
  Write-Host ">>> python $($Args -join ' ')" -ForegroundColor Cyan
  & python @Args
  if ($LASTEXITCODE -ne 0) { throw "Python step failed: $($Args -join ' ')" }
}

# --- Paths, logs, env ---
$RepoRoot = (Resolve-Path ".").Path
$env:PYTHONPATH = "$RepoRoot;$($env:PYTHONPATH)"

$LogsDir = Ensure-Dir (Join-Path $RepoRoot "logs")
$OutRoot = Ensure-Dir $OutRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "phase3_isic_$ts.log"
Start-Transcript -LiteralPath $LogFile | Out-Null

try {
  # Sanity
  $ISICRoot     = Resolve-Existing $ISICRoot
  $TeacherCkpt  = Resolve-Ckpt $TeacherCkpt
  $LabelMap     = Resolve-Existing $LabelMap
  $IsicToHamMap = Resolve-Existing $IsicToHamMap -AllowEmpty

  $robArg = @()
  if ($DoRobustness) { $robArg += @("--do-robust", "--levels", $CorruptionLevels) }

  # -----------------------------
  # 1) Teacher eval (HAM mode)
  # -----------------------------
  Write-Host "`n[1/2] Evaluate Teacher (HAM-mode)" -ForegroundColor Yellow
  $TeacherOut = Ensure-Dir (Join-Path $OutRoot "teacher_isic50")

  $baseArgs = @(
    "-m", "external_src.eval.eval_pack_isic",
    "--mode", "ham",
    "--dataset-root", $ISICRoot,
    "--arch", "resnet50",
    "--ckpt", $TeacherCkpt,
    "--outdir", $TeacherOut,
    "--label-map", $LabelMap,
    "--eval-batch-size", $EvalBatchSize,
    "--img-size", $ImgSize,
    "--device", $Device,
    "--bins", $Bins,
    "--lat-gpu-repeats", $LatGPURepeats,
    "--lat-gpu-warmup",  $LatGPUWarmup,
    "--lat-gpu-batches", $LatGPUBatches,
    "--lat-cpu-repeats", $LatCPURepeats,
    "--lat-cpu-warmup",  $LatCPUWarmup,
    "--lat-cpu-batches", $LatCPUBatches,
    "--num-workers", $NumWorkers
  )
  if ($IsicToHamMap -ne "") { $baseArgs += @("--isic-to-ham-map", $IsicToHamMap) }
  Py @baseArgs @robArg

  # -----------------------------
  # 2) Students eval (HAM mode)
  # -----------------------------
  Write-Host "`n[2/2] Evaluate Students (HAM-mode)" -ForegroundColor Yellow
  foreach ($item in $Students) {
    $parts = $item.Split("|")
    if ($parts.Count -lt 3) { throw "Students entry must be 'arch|ckpt_or_dir|tag' : $item" }
    $arch = $parts[0].Trim()
    $ckpt = Resolve-Ckpt ($parts[1].Trim())
    $tag  = $parts[2].Trim()

    $OutDir = Ensure-Dir (Join-Path $OutRoot $tag)
    Write-Host (" - " + $tag + " (" + $arch + ")") -ForegroundColor DarkCyan

    $argsStu = @(
      "-m", "external_src.eval.eval_pack_isic",
      "--mode", "ham",
      "--dataset-root", $ISICRoot,
      "--arch", $arch,
      "--ckpt", $ckpt,
      "--outdir", $OutDir,
      "--label-map", $LabelMap,
      "--eval-batch-size", $EvalBatchSize,
      "--img-size", $ImgSize,
      "--device", $Device,
      "--bins", $Bins,
      "--lat-gpu-repeats", $LatGPURepeats,
      "--lat-gpu-warmup",  $LatGPUWarmup,
      "--lat-gpu-batches", $LatGPUBatches,
      "--lat-cpu-repeats", $LatCPURepeats,
      "--lat-cpu-warmup",  $LatCPUWarmup,
      "--lat-cpu-batches", $LatCPUBatches,
      "--num-workers", $NumWorkers
    )
    if ($IsicToHamMap -ne "") { $argsStu += @("--isic-to-ham-map", $IsicToHamMap) }
    Py @argsStu @robArg
  }

  Write-Host "`nDone. Artifacts root:" -ForegroundColor Green
  Write-Host (" - " + $OutRoot)

} catch {
  Write-Error $_.Exception.Message
  throw
} finally {
  Stop-Transcript | Out-Null
  Write-Host "Log saved to: $LogFile" -ForegroundColor DarkGray
}
