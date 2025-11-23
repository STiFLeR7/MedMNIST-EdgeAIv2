# D:\MedMNIST-EdgeAIv2\generate_report.ps1
# -----------------------------------------------------------------------------
# Summarize all KD results under RESULTS/teachers and RESULTS/students
# Outputs KD_Report_<date>.csv in RESULTS/
# -----------------------------------------------------------------------------

$root = "D:\MedMNIST-EdgeAIv2\RESULTS"
$outFile = Join-Path $root ("KD_Report_{0}.csv" -f (Get-Date -Format "yyyyMMdd_HHmm"))
$allRows = @()

function Get-CsvRows($csvPath, $title) {
    try {
        $csv = Import-Csv -Path $csvPath
        foreach ($row in $csv) {
            [PSCustomObject]@{
                Title   = $title
                Metric  = $row.metric
                Value   = $row.value
                Source  = $csvPath
            }
        }
    } catch {
        Write-Warning "Failed to read $csvPath"
    }
}

# 1. Teacher results
$teacherDirs = Get-ChildItem "$root" -Directory | Where-Object { $_.Name -like "teacher_*" }
foreach ($t in $teacherDirs) {
    $tables = Join-Path $t.FullName "tables"
    if (Test-Path $tables) {
        $csvs = Get-ChildItem $tables -Filter "summary_*.csv" -File -Recurse
        foreach ($csv in $csvs) {
            $title = "Teacher | $($t.Name) | $(Split-Path $csv.BaseName -Leaf)"
            $allRows += Get-CsvRows $csv.FullName $title
        }
    }
}

# 2. Student results (ResNet18, MobileNetV2, EfficientNet-B0)
$studentRoot = Join-Path $root "students"
$students = Get-ChildItem $studentRoot -Directory
foreach ($stu in $students) {
    $stuName = $stu.Name
    $abls = Get-ChildItem $stu.FullName -Directory
    foreach ($abl in $abls) {
        $ablName = $abl.Name
        $datasets = Get-ChildItem $abl.FullName -Directory
        foreach ($ds in $datasets) {
            $tables = Join-Path $ds.FullName "tables"
            if (Test-Path $tables) {
                $csvs = Get-ChildItem $tables -Filter "summary_*.csv" -File -Recurse
                foreach ($csv in $csvs) {
                    $title = "Student | $stuName | Ablation=$ablName | Dataset=$($ds.Name)"
                    $allRows += Get-CsvRows $csv.FullName $title
                }
            }
        }
    }
}

# 3. Export consolidated CSV
if ($allRows.Count -gt 0) {
    $allRows | Export-Csv -Path $outFile -NoTypeInformation -Encoding UTF8
    Write-Host "`n✅ Report generated:" $outFile
    Write-Host "Total entries:" $allRows.Count
} else {
    Write-Warning "No summary_*.csv files found under $root"
}
