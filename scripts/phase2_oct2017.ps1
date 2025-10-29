<#
Phase 2 Runner — OCT2017 end-to-end evidence
Usage (from repo root):
  pwsh -File .\scripts\phase2_oct2017.ps1 `
    -DataRoot ".\data\OCT2017" `
    -TeacherCkpt ".\models\teachers\oct2017_resnet50_seed0\ckpt-best.pth" `
    -Students "efficientnet_b0|.\models\students\oct2017_effb0_kdat_seed0\ckpt-best.pth|student_effb0",
              "mobilenet_v2|.\models\students\oct2017_mbv2_kdat_seed0\ckpt-best.pth|student_mbv2",
              "resnet18|.\models\students\oct2017_resnet18_kdat_seed0\ckpt-best.pth|student_resnet18" `
    -OutRoot ".\reports\phase2_oct2017" `
    -EvalBatchSize 32 `
    -Device cuda `
    -DoRobustness:$true
#>

[CmdletBinding()]
param(
  [string]$DataRoot = ".\data\OCT2017",
  [string]$TeacherCkpt = ".\models\teachers\oct2017_resnet50_seed0\ckpt-best.pth",

  # Students list entries: "arch|ckpt|tag"
  [string[]]$Students = @(
    "efficientnet_b0|.\models\students\oct2017_effb0_kdat_seed0\ckpt-best.pth|student_effb0",
    "mobilenet_v2|.\models\students\oct2017_mbv2_kdat_seed0\ckpt-best.pth|student_mbv2",
    "resnet18|.\models\students\oct2017_resnet18_kdat_seed0\ckpt-best.pth|student_resnet18"
  ),

  [string]$OutRoot = ".\reports\phase2_oct2017",
  [ValidateSet("cuda","cpu")] [string]$Device = "cuda",
  [int]$EvalBatchSize = 32,
  [int]$ImgSize = 224,
  [int]$Bins = 15,

  # Robustness knobs
  [switch]$DoRobustness = $true,
  [string]$CorruptionLevels = "gauss:0.1,0.2,0.3;jpeg:90,70,50;contrast:0.8,0.6",

  # Efficiency knobs
  [int]$LatGPURepeats = 100,
  [int]$LatGPUWarmup = 20,
  [string]$LatGPUBatches = "1,2,4,8",
  [int]$LatCPURepeats = 200,
  [int]$LatCPUWarmup = 30,
  [string]$LatCPUBatches = "1,2,4",

  [int]$NumWorkers = 4
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
function Resolve-Existing([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { throw "Path not found: $p" }
  return (Resolve-Path -LiteralPath $p).Path
}
function Py() {
  param([Parameter(ValueFromRemainingArguments=$true)] [string[]]$Args)
  Write-Host ">>> python $($Args -join ' ')" -ForegroundColor Cyan
  & python @Args
  if ($LASTEXITCODE -ne 0) { throw "Python step failed: $($Args -join ' ')" }
}

# --- Paths, Logs, PYTHONPATH ---
$RepoRoot = (Resolve-Path ".").Path
$env:PYTHONPATH = "$RepoRoot;$($env:PYTHONPATH)"

$LogsDir = Ensure-Dir (Join-Path $RepoRoot "logs")
$OutRoot = Ensure-Dir $OutRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "phase2_oct2017_$ts.log"
Start-Transcript -LiteralPath $LogFile | Out-Null

try {
  # Sanity check inputs
  $DataRoot   = Resolve-Existing $DataRoot
  $TeacherCkpt = Resolve-Existing $TeacherCkpt

  # -----------------------------
  # 1) Teacher eval (ResNet50)
  # -----------------------------
  Write-Host "`n[1/3] Evaluate Teacher (ResNet50)" -ForegroundColor Yellow
  $TeacherOut = Ensure-Dir (Join-Path $OutRoot "teacher_resnet50")

  $robArg = @()
  if ($DoRobustness) { $robArg += @("--do-robust", "--levels", $CorruptionLevels) }

  Py -m external_src.eval.eval_pack_oct2017 `
    --dataset-root $DataRoot `
    --arch resnet50 `
    --ckpt $TeacherCkpt `
    --outdir $TeacherOut `
    --eval-batch-size $EvalBatchSize `
    --img-size $ImgSize `
    --device $Device `
    --bins $Bins `
    --lat-gpu-repeats $LatGPURepeats `
    --lat-gpu-warmup $LatGPUWarmup `
    --lat-gpu-batches $LatGPUBatches `
    --lat-cpu-repeats $LatCPURepeats `
    --lat-cpu-warmup $LatCPUWarmup `
    --lat-cpu-batches $LatCPUBatches `
    --num-workers $NumWorkers `
    @robArg

  # -----------------------------
  # 2) Students eval (loop)
  # -----------------------------
  Write-Host "`n[2/3] Evaluate Students" -ForegroundColor Yellow
  $StudentOutDirs = @()
  foreach ($item in $Students) {
    $parts = $item.Split("|")
    if ($parts.Count -lt 3) { throw "Students entry must be 'arch|ckpt|tag' : $item" }
    $arch = $parts[0].Trim()
    $ckpt = Resolve-Existing $parts[1].Trim()
    $tag  = $parts[2].Trim()

    $OutDir = Ensure-Dir (Join-Path $OutRoot $tag)
    $StudentOutDirs += $OutDir

    Write-Host (" - " + $tag + " (" + $arch + ")") -ForegroundColor DarkCyan

    Py -m external_src.eval.eval_pack_oct2017 `
      --dataset-root $DataRoot `
      --arch $arch `
      --ckpt $ckpt `
      --outdir $OutDir `
      --eval-batch-size $EvalBatchSize `
      --img-size $ImgSize `
      --device $Device `
      --bins $Bins `
      --lat-gpu-repeats $LatGPURepeats `
      --lat-gpu-warmup $LatGPUWarmup `
      --lat-gpu-batches $LatGPUBatches `
      --lat-cpu-repeats $LatCPURepeats `
      --lat-cpu-warmup $LatCPUWarmup `
      --lat-cpu-batches $LatCPUBatches `
      --num-workers $NumWorkers `
      @robArg
  }

  # -----------------------------
  # 3) Pack consolidated PDF
  # -----------------------------
  Write-Host "`n[3/3] Pack Phase-2 OCT2017 report" -ForegroundColor Yellow
  $StudentDirsCSV = ($StudentOutDirs -join ",")

  if (Test-Path ".\external_src\report\pack_phase2_oct2017.py") {
    Py -m external_src.report.pack_phase2_oct2017 `
      --teacher-dir $TeacherOut `
      --students-dirs $StudentDirsCSV `
      --out (Join-Path $OutRoot "phase2_consolidated.pdf") `
      --tb-root ".\models"
  } else {
    Write-Host "Pack script not found (external_src/report/pack_phase2_oct2017.py). Skipping PDF." -ForegroundColor DarkYellow
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
