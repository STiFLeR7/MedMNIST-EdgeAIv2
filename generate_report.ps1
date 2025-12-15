# D:\MedMNIST-EdgeAIv2\generate_report.ps1
# ---------------------------------------------------------------------------
# Generate KD report (Markdown + optional PDF) from RESULTS/
# - Aggregates summary_*.csv metrics
# - Embeds confusion matrix images (confmat_*.png / confmat_norm_*.png)
# - Saves KD_Report_<timestamp>.md and optionally KD_Report_<timestamp>.pdf
# - Optional LLM hook: D:/FastFare/models/llama-2-7b-chat-hf-q4_k_m.gguf
# ---------------------------------------------------------------------------

Set-StrictMode -Version Latest
$root = "D:\MedMNIST-EdgeAIv2\RESULTS"
if (-not (Test-Path $root)) { Write-Error "RESULTS folder not found at $root"; exit 1 }

$ts = Get-Date -Format "yyyyMMdd_HHmm"
$outMd  = Join-Path $root ("KD_Report_{0}.md" -f $ts)
$outPdf = Join-Path $root ("KD_Report_{0}.pdf" -f $ts)

# Optional LLM model file path (user-provided)
$llmPath = "D:/FastFare/models/llama-2-7b-chat-hf-q4_k_m.gguf"

# Helpers
function Append-Line([string]$text) {
    Add-Content -Path $outMd -Value $text
}
function Safe-ImportCsv($csvPath) {
    try {
        return Import-Csv -Path $csvPath -ErrorAction Stop
    } catch {
        Write-Warning "Could not import CSV: $csvPath"
        return $null
    }
}

# Start fresh
if (Test-Path $outMd) { Remove-Item $outMd -Force }
Append-Line "# KD Results Report"
Append-Line ""
Append-Line "> Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Append-Line ""
Append-Line "---"
Append-Line ""

# gather summary function
function Add-Run-Section($type, $rootFolder, $titlePrefix) {
    # type: "Teacher" or "Student"
    # rootFolder: full path to run folder that contains a 'tables' subfolder
    # titlePrefix: textual title to show (model/ablation/dataset context)
    $tablesDir = Join-Path $rootFolder "tables"
    $figsDir   = Join-Path $rootFolder "figs"
    if (-not (Test-Path $tablesDir)) { return }

    # read summary_*.csv (there may be multiple)
    $summaryFiles = Get-ChildItem -Path $tablesDir -Filter "summary_*.csv" -File -ErrorAction SilentlyContinue
    if ($summaryFiles.Count -eq 0) {
        # maybe older naming; fallback to any csv
        $summaryFiles = Get-ChildItem -Path $tablesDir -Filter "*.csv" -File -ErrorAction SilentlyContinue
    }

    foreach ($sf in $summaryFiles) {
        $title = "$titlePrefix - Summary: $($sf.BaseName)"
        Append-Line "## $title"
        Append-Line ""
        $csv = Safe-ImportCsv $sf.FullName
        if ($csv -ne $null) {
            Append-Line "| Metric | Value | Source |"
            Append-Line "|---|---:|---|"
            foreach ($r in $csv) {
                $metric = $r.metric
                $value  = $r.value
                Append-Line ("| {0} | {1} | `{2}` |" -f $metric, $value, $sf.FullName)
            }
            Append-Line ""
        }

        # per-class breakdown if exists
        $pc = Join-Path $tablesDir ("perclass_{0}.csv" -f $sf.BaseName.Replace("summary_",""))
        if (-not (Test-Path $pc)) {
            # try any perclass file
            $pc = Get-ChildItem -Path $tablesDir -Filter "perclass_*.csv" -File -ErrorAction SilentlyContinue | Select-Object -First 1
        } else {
            $pc = Get-Item -Path $pc -ErrorAction SilentlyContinue
        }
        if ($pc) {
            Append-Line "### Per-class metrics (CSV: `$(Split-Path $pc -Leaf)`)"
            Append-Line ""
            Append-Line "CSV: `$($pc.FullName)`"
            Append-Line ""
        }

        # confusion matrices in figs/
        if (Test-Path $figsDir) {
            $cmFiles = Get-ChildItem -Path $figsDir -Include "confmat_*.png","confmat_norm_*.png" -File -Recurse -ErrorAction SilentlyContinue
            if ($cmFiles.Count -gt 0) {
                Append-Line "### Confusion matrices"
                Append-Line ""
                foreach ($img in $cmFiles) {
                    # Use relative path if under $root; otherwise absolute
                    $rel = $img.FullName
                    if ($rel.StartsWith($root)) {
                        $rel = $rel.Substring($root.Length).TrimStart('\','/')
                        $rel = ".\" + (Join-Path "RESULTS" $rel.Replace("/","\")) # ensure windows-style
                    }
                    Append-Line "#### `$(Split-Path $img.FullName -Leaf)`"
                    Append-Line ""
                    Append-Line "![confmat]($rel)"
                    Append-Line ""
                }
            }
        }

        Append-Line "---"
        Append-Line ""
    }
}

# 1) Teacher directories
$teacherDirs = Get-ChildItem -Path $root -Directory | Where-Object { $_.Name -like "teacher_*" }
if ($teacherDirs.Count -gt 0) {
    Append-Line "## Teachers"
    Append-Line ""
    foreach ($t in $teacherDirs) {
        $title = "Teacher | $($t.Name)"
        Append-Line "### $title"
        Append-Line ""
        Add-Run-Section -type "Teacher" -rootFolder $t.FullName -titlePrefix $title
    }
} else {
    Append-Line "## Teachers"
    Append-Line ""
    Append-Line "_No teacher directories found under RESULTS/_"
    Append-Line ""
}

# 2) Students (students/<model>/<ablation>/<dataset>/)
$studentsRoot = Join-Path $root "students"
if (Test-Path $studentsRoot) {
    Append-Line "## Students"
    Append-Line ""
    $students = Get-ChildItem -Path $studentsRoot -Directory -ErrorAction SilentlyContinue
    foreach ($stu in $students) {
        Append-Line "### Student model: $($stu.Name)"
        Append-Line ""
        $ablations = Get-ChildItem -Path $stu.FullName -Directory -ErrorAction SilentlyContinue
        foreach ($abl in $ablations) {
            Append-Line "#### Ablation: $($abl.Name)"
            Append-Line ""
            $datasets = Get-ChildItem -Path $abl.FullName -Directory -ErrorAction SilentlyContinue
            foreach ($ds in $datasets) {
                $title = "Student | $($stu.Name) | Ablation=$($abl.Name) | Dataset=$($ds.Name)"
                Append-Line "##### $title"
                Append-Line ""
                Add-Run-Section -type "Student" -rootFolder $ds.FullName -titlePrefix $title
            }
        }
    }
} else {
    Append-Line "## Students"
    Append-Line ""
    Append-Line "_No students folder found under RESULTS/students/_"
    Append-Line ""
}

# 3) Global summary table (collect best accuracy per run if present)
Append-Line "## Consolidated Summary"
Append-Line ""
$summaryRows = @()
# find all summary CSVs under RESULTS
$allSummaryFiles = Get-ChildItem -Path $root -Filter "summary_*.csv" -File -Recurse -ErrorAction SilentlyContinue
foreach ($sf in $allSummaryFiles) {
    $csv = Safe-ImportCsv $sf.FullName
    if ($csv -ne $null) {
        # look for accuracy row
        foreach ($r in $csv) {
            $m = $r.metric.ToLower()
            if ($m -like "*accuracy*" -or $m -eq "accuracy") {
                $entry = [PSCustomObject]@{
                    Source = $sf.FullName
                    Accuracy = [double]$r.value
                    MetricFile = $sf.FullName
                }
                $summaryRows += $entry
                break
            }
        }
    }
}

if ($summaryRows.Count -gt 0) {
    Append-Line "| Source | Accuracy |"
    Append-Line "|---|---:|"
    foreach ($s in $summaryRows | Sort-Object -Property Accuracy -Descending) {
        Append-Line ("| `{0}` | {1} |" -f $s.Source, $s.Accuracy)
    }
} else {
    Append-Line "_No accuracy metrics found in summary_*.csv files._"
}
Append-Line ""
Append-Line "---"
Append-Line ""

# 4) Optional: Use a local LLM to generate a theory-based addendum (placeholder)
if (Test-Path $llmPath) {
    Append-Line "## LLM-generated Theory Addendum (optional)"
    Append-Line ""
    Append-Line "_Local GGUF model found at_: `"$llmPath`\""
    Append-Line ""
    Append-Line "_If you want to auto-generate a high-level narrative / theory PDF using the local LLM, uncomment and customize the LLM invocation block in this script._"
    Append-Line ""
    Append-Line "Notes: Example invocation (not enabled):"
    Append-Line "```powershell"
    Append-Line "# Example: run a local python tool that loads the GGUF model and creates 'llm_addendum.md'"
    Append-Line "# python tools/local_llm_generate.py --model `"$llmPath`" --input `"$outMd`" --output `"$root\\llm_addendum.md`""
    Append-Line "```"
    Append-Line ""
}

# 5) Export note and attempt to convert to PDF via pandoc (if present)
Append-Line "## Files produced"
Append-Line ""
Append-Line "- Markdown report: `$(Split-Path $outMd -Leaf)`"
Append-Line "- PDF (if pandoc available): `$(Split-Path $outPdf -Leaf)`"
Append-Line ""
Append-Line "---"
Append-Line ""
Append-Line "_End of report generation._"

Write-Host "`n✅ Markdown report written to: $outMd`n"

# Try to convert to PDF with pandoc (if available)
$hasPandoc = $false
try {
    $p = Get-Command pandoc -ErrorAction SilentlyContinue
    if ($p) { $hasPandoc = $true }
} catch { $hasPandoc = $false }

if ($hasPandoc) {
    Write-Host "Pandoc found. Attempting to generate PDF..."
    # use a simple CSS for nicer look if desired
    $css = Join-Path $root "report_style.css"
    if (-not (Test-Path $css)) {
        @"
body { font-family: Arial, Helvetica, sans-serif; line-height: 1.4; }
h1,h2,h3,h4 { color: #1f4e79; }
table { border-collapse: collapse; width: 100%; }
table th, table td { border: 1px solid #ddd; padding: 6px; }
"@ | Out-File -FilePath $css -Encoding UTF8
    }
    $pandocArgs = @(
        "--from", "markdown",
        "--pdf-engine", "wkhtmltopdf",  # fallback to wkhtmltopdf; pandoc will try default if not present
        "--css", $css,
        "-o", $outPdf,
        $outMd
    )
    try {
        & pandoc @pandocArgs
        if (Test-Path $outPdf) {
            Write-Host "✅ PDF generated: $outPdf"
        } else {
            Write-Warning "Pandoc ran but PDF not found. Check pandoc/wkhtmltopdf installation."
        }
    } catch {
        Write-Warning "Pandoc conversion failed: $($_.Exception.Message)"
    }
} else {
    Write-Warning "Pandoc not found on PATH. To create a PDF, install pandoc and wkhtmltopdf, or run the Markdown through your preferred converter."
    Write-Host "You can open the markdown: $outMd"
}

# Final message
Write-Host "`nReport generation complete. Markdown -> $outMd"
if (Test-Path $outPdf) { Write-Host "PDF -> $outPdf" } else { Write-Host "PDF not generated (pandoc missing or conversion failed)." }
