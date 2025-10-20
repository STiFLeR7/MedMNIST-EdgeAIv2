[CmdletBinding()]
param(
  [string]$Prefix = "ham10000",
  [string]$DataRoot = ".\data\HAM10000",
  [string]$TeacherRoot = ".\models\teachers\runs_ham10000_resnet50",
  [string]$StudentRoots = ".\models\students\distilled_resnet18_ham10000,.\models\students\distilled_mobilenetv2_ham10000,.\models\students\distilled_efficientnetb0_ham10000",
  [switch]$SynthesizeCurves = $true
)

$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
$pyArgs = @(
  "-m","external_src.eval.make_tfevents_from_artifacts",
  "--prefix",$Prefix,
  "--repo-root",$repo,
  "--data-root",$DataRoot,
  "--teacher-root",$TeacherRoot,
  "--student-roots",$StudentRoots
)
if ($SynthesizeCurves) { $pyArgs += "--synthesize-curves" }

Write-Host ">>> python $($pyArgs -join ' ')" -ForegroundColor Cyan
& python @pyArgs
if ($LASTEXITCODE -ne 0) { throw "TB event generation failed." }

Write-Host "TensorBoard ready. Try:" -ForegroundColor Green
Write-Host "  tensorboard --logdir .\models" -ForegroundColor Yellow
