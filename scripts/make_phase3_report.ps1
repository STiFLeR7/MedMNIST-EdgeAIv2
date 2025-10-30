<#
Phase 3 Report Maker — ISIC (HAM-mode)
- Packs teacher + students evidence (produced by phase3_isic.ps1) into a single A4 PDF.
- Accepts Students/TBRoots as either PowerShell arrays or single comma-separated strings.

Usage (from repo root):
  pwsh -File .\scripts\make_phase3_report.ps1 `
    -ISICRoot ".\data\ISIC" `
    -OutRoot ".\reports\phase3_isic" `
    -Students "student_effb0_isic","student_mbv2_isic","student_resnet18_isic" `
    -TBRoots ".\models\teachers\isic_resnet50_v2\tb",".\models\students\isic_kdat_all\tb_global" `
    -OutPdf ".\reports\phase3_isic\phase3_consolidated.pdf"
#>

[CmdletBinding()]
param(
  [string]$ISICRoot = ".\data\ISIC",
  [string]$OutRoot  = ".\reports\phase3_isic",

  # Can be array: "a","b","c"  OR a single string: "a,b,c"
  [object]$Students = @(),

  # Can be array or single comma-separated string
  [object]$TBRoots = @(),

  [string]$OutPdf = ".\reports\phase3_isic\phase3_consolidated.pdf",
  [int]$SamplesPerClass = 4,
  [int]$ImgSize = 224,
  [string]$TeacherFolder = "teacher_isic50"
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
function Normalize-List([object]$InputVal) {
  # Accepts array or single string; returns trimmed string[]
  if ($null -eq $InputVal) { return @() }
  if ($InputVal -is [System.Array]) {
    return @($InputVal | ForEach-Object { ($_ | Out-String).Trim().Trim('"').Trim() } | Where-Object { $_ -ne "" })
  }
  # Single object -> string; split by comma
  $s = ($InputVal | Out-String).Trim()
  if ([string]::IsNullOrWhiteSpace($s)) { return @() }
  return @($s.Split(",") | ForEach-Object { $_.Trim().Trim('"').Trim() } | Where-Object { $_ -ne "" })
}

# --- Paths, logs, PYTHONPATH ---
$RepoRoot = (Resolve-Path ".").Path
$env:PYTHONPATH = "$RepoRoot;$($env:PYTHONPATH)"

$LogsDir = Ensure-Dir (Join-Path $RepoRoot "logs")
$OutRoot = Ensure-Dir $OutRoot
$ISICRoot = Resolve-Existing $ISICRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "make_phase3_isic_$ts.log"
Start-Transcript -LiteralPath $LogFile | Out-Null

try {
  $TeacherDir = Resolve-Existing (Join-Path $OutRoot $TeacherFolder)

  # Normalize inputs (accept array OR single comma-separated string)
  $StudentsList = Normalize-List $Students
  $TBRootsList  = Normalize-List $TBRoots

  # Collect student result dirs
  $StudentDirs = @()
  if ($StudentsList.Count -gt 0) {
    foreach ($tag in $StudentsList) {
      $d = Join-Path $OutRoot $tag
      $StudentDirs += (Resolve-Existing $d)
    }
  } else {
    # Auto-discover any dirs under OutRoot except the teacher dir
    Get-ChildItem -LiteralPath $OutRoot -Directory | ForEach-Object {
      if ($_.FullName -ne $TeacherDir) {
        $StudentDirs += $_.FullName
      }
    }
    if ($StudentDirs.Count -eq 0) {
      throw "No student result directories found under $OutRoot. Provide -Students or run phase3_isic.ps1 first."
    }
  }

  # Resolve TB roots (optional)
  $tbList = @()
  foreach ($t in $TBRootsList) {
    if ([string]::IsNullOrWhiteSpace($t)) { continue }
    if (Test-Path -LiteralPath $t) {
      $tbList += (Resolve-Existing $t)
    } else {
      Write-Warning "TB root not found: $t (skipping)"
    }
  }

  $outPdfParent = Split-Path -Parent $OutPdf
  if (-not [string]::IsNullOrWhiteSpace($outPdfParent)) {
    Ensure-Dir $outPdfParent | Out-Null
  }

  $studentsCsv = ($StudentDirs -join ",")
  $tbCsv = ($tbList -join ",")

  Py -m external_src.report.pack_phase3_isic `
    --dataset-root $ISICRoot `
    --teacher-dir $TeacherDir `
    --students-dirs $studentsCsv `
    --tb-root-list $tbCsv `
    --out $OutPdf `
    --title "MedMNIST-EdgeAI v2 — Phase-3 ISIC (HAM-mode)" `
    --samples-per-class $SamplesPerClass `
    --img-size $ImgSize

  Write-Host "`nWrote PDF:" -ForegroundColor Green
  Write-Host (" - " + (Resolve-Existing $OutPdf))

} catch {
  Write-Error $_.Exception.Message
  throw
} finally {
  Stop-Transcript | Out-Null
  Write-Host "Log saved to: $LogFile" -ForegroundColor DarkGray
}
