<# 
Phase 1 Runner — HAM10000 end-to-end evidence
Usage (from repo root):
  pwsh -File .\scripts\phase1_ham10000.ps1 `
    -DataRoot ".\data\HAM10000" `
    -SplitsDir ".\external_data\splits\HAM10000" `
    -Seeds "0,1,2,3,4" `
    -StudentRoots ".\models\students\distilled_resnet18_ham10000,.\\models\\students\\distilled_mobilenetv2_ham10000,.\\models\\students\\distilled_efficientnetb0_ham10000" `
    -TeacherRoot ".\models\teachers\runs_ham10000_resnet50" `
    -DoRobustness:$true -Device "cuda"

Notes:
- Assumes the Python files you pasted earlier are present under external_src/, and that external_src/* folders contain __init__.py.
- If CUDA is not available, run with -Device "cpu" (GPU latency steps are skipped).
- Robustness needs a split JSON (e.g., seed_0) because it evaluates images directly.
#>

[CmdletBinding()]
param(
  [string]$DataRoot = ".\data\HAM10000",
  [string]$SplitsDir = ".\external_data\splits\HAM10000",
  [string]$Seeds = "0,1,2,3,4",
  # Comma-separated list of student model roots (each root contains seed_* subdirs)
  [string]$StudentRoots = ".\models\students\distilled_resnet18_ham10000,.\models\students\distilled_mobilenetv2_ham10000,.\models\students\distilled_efficientnetb0_ham10000",
  [string]$TeacherRoot = ".\models\teachers\runs_ham10000_resnet50",
  [ValidateSet("cuda","cpu")] [string]$Device = "cuda",
  [int]$CalibBins = 15,
  [switch]$DoRobustness = $true,
  [string]$CorruptionLevels = "gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6",
  [int]$LatencyRepeatsGPU = 100,
  [int]$LatencyWarmupGPU = 20,
  [int]$LatencyRepeatsCPU = 200,
  [int]$LatencyWarmupCPU = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
  $dir = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
  if (-not $dir) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Resolve-Existing([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { throw "Path not found: $p" }
  return (Resolve-Path -LiteralPath $p).Path
}

function Py() {
  # Wrapper to invoke python with proper error bubbling
  param([Parameter(ValueFromRemainingArguments=$true)] [string[]]$Args)
  Write-Host ">>> python $($Args -join ' ')" -ForegroundColor Cyan
  & python @Args
  if ($LASTEXITCODE -ne 0) { throw "Python step failed: $($Args -join ' ')" }
}

# --- Paths ---
$RepoRoot = Resolve-Path "."
$TablesCore         = Join-Path $RepoRoot "tables\core"
$TablesCalib        = Join-Path $RepoRoot "tables\calibration"
$TablesRobust       = Join-Path $RepoRoot "tables\robustness"
$TablesEff          = Join-Path $RepoRoot "tables\efficiency"
$TablesAbl          = Join-Path $RepoRoot "tables\ablations"
$FigsCalib          = Join-Path $RepoRoot "figs\calibration"
$FigsRobust         = Join-Path $RepoRoot "figs\robustness"
$FigsPareto         = Join-Path $RepoRoot "figs\pareto"
$LogsDir            = Join-Path $RepoRoot "logs"

Ensure-Dir $TablesCore; Ensure-Dir $TablesCalib; Ensure-Dir $TablesRobust
Ensure-Dir $TablesEff;  Ensure-Dir $TablesAbl;   Ensure-Dir $FigsCalib
Ensure-Dir $FigsRobust; Ensure-Dir $FigsPareto;  Ensure-Dir $LogsDir

# --- Logging ---
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "phase1_ham10000_$ts.log"
Start-Transcript -LiteralPath $LogFile | Out-Null

try {
  # --- Sanity inputs ---
  $DataRoot = Resolve-Existing $DataRoot
  $SplitsDir = Resolve-Existing $SplitsDir
  $TeacherRoot = Resolve-Existing $TeacherRoot

  $StudentRootList = @()
  foreach ($r in $StudentRoots.Split(",")) {
    $rp = Resolve-Existing $r.Trim()
    $StudentRootList += $rp
  }

  $SeedList = @()
  foreach ($s in $Seeds.Split(",")) { $SeedList += [int]$s }

  # --- Helper: Compose globs from roots ---
  function Glob-StudentPreds {
    param([string]$Root)
    # Expect seed_*/test_preds.parquet
    return (Join-Path $Root "seed_*" "test_preds.parquet")
  }
  function Glob-StudentCkpts {
    param([string]$Root)
    # Expect seed_*/best.pt (or .ts)
    return (Join-Path $Root "seed_*" "best.pt")
  }

  # =========================
  # 1) AGGREGATE CI (Teacher + Students)
  # =========================
  Write-Host "`n[1/6] Aggregate CI — teacher + students" -ForegroundColor Yellow

  $TeacherOut = Join-Path $TablesCore "ham10000_teacher_ci.csv"
  Py -m external_src.eval.aggregate_runs `
    --roots $TeacherRoot `
    --task derma `
    --n_boot 2000 `
    --out $TeacherOut

  $StudentRootsCSV = ($StudentRootList -join ",")
  $StudentsOut = Join-Path $TablesCore "ham10000_students_ci.csv"
  Py -m external_src.eval.aggregate_runs `
    --roots $StudentRootsCSV `
    --task derma `
    --n_boot 2000 `
    --out $StudentsOut

  # =========================
  # 2) CALIBRATION + RELIABILITY + OPERATING POINT
  # =========================
  Write-Host "`n[2/6] Calibration metrics, reliability curves, operating points" -ForegroundColor Yellow

  # Build a combined glob union for all student preds
  $PredGlobs = @()
  foreach ($sr in $StudentRootList) { $PredGlobs += (Glob-StudentPreds -Root $sr) }
  $PredGlobUnion = ($PredGlobs -join ";")  # io_utils scans each pattern separately

  $CalibOut = Join-Path $TablesCalib "ham10000_calibration_metrics.csv"
  Py -m external_src.eval.calibration_metrics `
    --preds-glob $PredGlobUnion `
    --out $CalibOut `
    --fit-temp `
    --bins $CalibBins `
    --adaptive

  Py -m external_src.eval.reliability_curves `
    --preds-glob $PredGlobUnion `
    --out $FigsCalib `
    --bins $CalibBins

  $OpOut = Join-Path $TablesCalib "ham10000_operating_points.csv"
  Py -m external_src.eval.operating_point `
    --preds-glob $PredGlobUnion `
    --metric macro_f1 `
    --out $OpOut

  # =========================
  # 3) ROBUSTNESS: CORRUPTIONS (optional toggle)
  # =========================
  if ($DoRobustness) {
    Write-Host "`n[3/6] Robustness — corruption suite" -ForegroundColor Yellow
    # Use seed_0 split by default
    $Seed0Split = Join-Path $SplitsDir "seed_0.json"
    if (-not (Test-Path -LiteralPath $Seed0Split)) {
      throw "Split file not found for robustness: $Seed0Split"
    }

    foreach ($sr in $StudentRootList) {
      $CkGlob = Glob-StudentCkpts -Root $sr
      $RobustCSV = Join-Path $TablesRobust "ham10000_corruptions_ci.csv"
      $RobustFig = Join-Path $FigsRobust  "ham10000_corruption_degradation.pdf"
      Py -m external_src.robustness.eval_corruptions `
        --pred-ckpt-glob $CkGlob `
        --data $DataRoot `
        --splits $Seed0Split `
        --levels $CorruptionLevels `
        --device ($Device) `
        --out $RobustCSV `
        --fig $RobustFig
    }
  } else {
    Write-Host "[3/6] Robustness — skipped" -ForegroundColor DarkYellow
  }

  # =========================
  # 4) EFFICIENCY: LATENCY (GPU+CPU) + MEMORY
  # =========================
  Write-Host "`n[4/6] Efficiency — latency + memory" -ForegroundColor Yellow
  # Build ckpt glob list
  $CkptGlobs = @()
  foreach ($sr in $StudentRootList) { $CkptGlobs += (Glob-StudentCkpts -Root $sr) }

  $LatGPUOut = Join-Path $TablesEff "ham10000_latency_gpu.csv"
  if ($Device -eq "cuda") {
    foreach ($g in $CkptGlobs) {
      Py -m external_src.efficiency.measure_latency `
        --ckpts $g `
        --device cuda `
        --repeats $LatencyRepeatsGPU `
        --warmup $LatencyWarmupGPU `
        --out $LatGPUOut
    }
  } else {
    Write-Host "CUDA not available or -Device cpu selected — skipping GPU latency." -ForegroundColor DarkYellow
  }

  $LatCPUOut = Join-Path $TablesEff "ham10000_latency_cpu.csv"
  foreach ($g in $CkptGlobs) {
    Py -m external_src.efficiency.measure_latency `
      --ckpts $g `
      --device cpu `
      --repeats $LatencyRepeatsCPU `
      --warmup $LatencyWarmupCPU `
      --out $LatCPUOut
  }

  $MemOut = Join-Path $TablesEff "ham10000_memory.csv"
  foreach ($g in $CkptGlobs) {
    Py -m external_src.efficiency.measure_memory `
      --ckpts $g `
      --out $MemOut
  }

  # =========================
  # 5) EFFICIENCY: PARETO MERGE + FIGURE
  # =========================
  Write-Host "`n[5/6] Efficiency — Pareto merge + figure" -ForegroundColor Yellow
  $ParetoTable = Join-Path $TablesEff "ham10000_pareto.csv"
  $ParetoFig   = Join-Path $FigsPareto "ham10000_acc_vs_latency.pdf"
  Py -m external_src.efficiency.pareto `
    --core $StudentsOut `
    --lat $LatGPUOut `
    --mem $MemOut `
    --out-table $ParetoTable `
    --out-fig $ParetoFig

  # =========================
  # 6) SUMMARY
  # =========================
  Write-Host "`n[6/6] Done. Artifacts:" -ForegroundColor Green
  @(
    $TeacherOut, $StudentsOut,
    $CalibOut, $OpOut,
    $ParetoTable, $ParetoFig
  ) | ForEach-Object { Write-Host (" - " + $_) }

} catch {
  Write-Error $_.Exception.Message
  throw
} finally {
  Stop-Transcript | Out-Null
  Write-Host "Log saved to: $LogFile" -ForegroundColor DarkGray
}
