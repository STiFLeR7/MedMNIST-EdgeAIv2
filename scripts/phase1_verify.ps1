<# Phase-1 Artifact Verifier — HAM10000 #>
[CmdletBinding()]
param(
  [string]$RepoRoot = ".",
  [string]$Prefix = "ham10000"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function To-Array($x) {
  if ($null -eq $x) { return @() }
  if ($x -is [System.Collections.IEnumerable] -and -not ($x -is [string])) { return @($x) }
  return @($x)
}

function Assert-Path([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { throw "Missing artifact: $p" }
}

function Get-CsvRows([string]$csvPath) {
  return (To-Array (Import-Csv -LiteralPath $csvPath))
}

function Get-CsvColumns([string]$csvPath) {
  $rows = Get-CsvRows -csvPath $csvPath
  $first = $rows | Select-Object -First 1
  if ($null -eq $first) { return @() }
  return @($first | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name)
}

function Assert-Columns([string]$csvPath, [string[]]$need) {
  $cols = Get-CsvColumns -csvPath $csvPath
  foreach ($c in $need) {
    if (-not ($cols -contains $c)) { throw "[$csvPath] missing column: $c" }
  }
}

function Assert-AnyColumn([string]$csvPath, [string[]]$candidates, [string]$what = "required column") {
  $cols = Get-CsvColumns -csvPath $csvPath
  foreach ($c in $candidates) {
    if ($cols -contains $c) { return }
  }
  throw "[$csvPath] none of the $what found: $($candidates -join ', ')"
}

function Get-NumericColumns([string]$csvPath, [string[]]$exclude) {
  $rows = Get-CsvRows -csvPath $csvPath
  if (-not $rows) { return @() }
  $cols = Get-CsvColumns -csvPath $csvPath
  $cands = $cols | Where-Object { $exclude -notcontains $_ }

  $numCols = @()
  foreach ($c in $cands) {
    $allNumeric = $true
    foreach ($r in ($rows | Select-Object -First 50)) {
      $v = $r.$c
      if ($null -eq $v -or $v -eq "") { continue }
      try { [void][double]$v } catch { $allNumeric = $false; break }
    }
    if ($allNumeric) { $numCols += $c }
  }
  return $numCols
}

$root = Resolve-Path -LiteralPath $RepoRoot
$tables = Join-Path $root "tables"
$figs   = Join-Path $root "figs"

$coreDir = Join-Path $tables "core"
$calDir  = Join-Path $tables "calibration"
$robDir  = Join-Path $tables "robustness"
$effDir  = Join-Path $tables "efficiency"

$paretoFig = Join-Path $figs "pareto\${Prefix}_acc_vs_latency.pdf"
$robFig    = Join-Path $figs "robustness\${Prefix}_corruption_degradation.pdf"

$teacherCI  = Join-Path $coreDir "${Prefix}_teacher_ci.csv"
$studentsCI = Join-Path $coreDir "${Prefix}_students_ci.csv"
$calMetrics = Join-Path $calDir  "${Prefix}_calibration_metrics.csv"
$opPoints   = Join-Path $calDir  "${Prefix}_operating_points.csv"
$robustCSV  = Join-Path $robDir  "${Prefix}_corruptions_ci.csv"
$latGPU     = Join-Path $effDir  "${Prefix}_latency_gpu.csv"
$latCPU     = Join-Path $effDir  "${Prefix}_latency_cpu.csv"
$memCSV     = Join-Path $effDir  "${Prefix}_memory.csv"
$paretoCSV  = Join-Path $effDir  "${Prefix}_pareto.csv"

# ---- Existence checks ----
$need = @(
  $teacherCI, $studentsCI,
  $calMetrics, $opPoints,
  $robustCSV,
  $latGPU, $latCPU, $memCSV,
  $paretoCSV,
  $paretoFig, $robFig
)
$need | ForEach-Object { Assert-Path $_ }

# ---- Minimal schema checks ----
Assert-Columns $teacherCI @("model")
Assert-Columns $studentsCI @("model")

# Calibration must have model + NLL + Brier + at least one ECE-like column
$needAnyECE = @("ece","ece_adaptive","ece_bin","ece_binwise","ece_ts")
Assert-Columns $calMetrics @("model","nll","brier")
Assert-AnyColumn $calMetrics $needAnyECE "ECE-like column"

# Operating points: require model + threshold-like + some metric column (named or any numeric)
Assert-Columns $opPoints @("model")
$thresholdCols = @("threshold","tau","cutoff","operating_point","op_point","t","decision_threshold")
Assert-AnyColumn $opPoints $thresholdCols "threshold-like column"

$metricNameCols = @("metric","metric_name","target_metric","metric_id")
$metricValueCols = @("macro_f1","f1","auroc","ap","accuracy","score","value")
$opCols = Get-CsvColumns -csvPath $opPoints
$hasNamedMetric = $false
foreach ($c in $metricNameCols + $metricValueCols) { if ($opCols -contains $c) { $hasNamedMetric = $true; break } }

if (-not $hasNamedMetric) {
  $exclude = @("model","seed") + $thresholdCols
  $numeric = Get-NumericColumns -csvPath $opPoints -exclude $exclude
  if (-not $numeric -or (To-Array $numeric).Length -lt 1) {
    throw "[$opPoints] no metric-like numeric column found. Columns present: $($opCols -join ', ')"
  }
}

Assert-Columns $robustCSV @("model","seed","macro_f1","tag")

# Efficiency: latency tables must have model/device/latency_ms
Assert-Columns $latGPU @("model","device","latency_ms")
Assert-Columns $latCPU @("model","device","latency_ms")

# Memory: params_mib required; peak_cuda_mib optional
Assert-Columns $memCSV @("model","params_mib")

# Pareto table basic sanity
Assert-Columns $paretoCSV @("model","score","latency_ms")

Write-Host "Phase-1 verification OK." -ForegroundColor Green
