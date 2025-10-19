# scripts/fetch_checkpoints.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Config
$projectRoot = (Resolve-Path "..\").ProviderPath  # if you run from scripts/, adjust as needed
# But if running from repo root: use $pwd
$projectRoot = Get-Location
$modelsDir = Join-Path $projectRoot "models"
$teachersDir = Join-Path $modelsDir "teachers"
$studentsDir = Join-Path $modelsDir "students"
$manifestPath = Join-Path $modelsDir "checkpoints-manifest.json"

# Official download URLs (torchvision / download.pytorch.org)
$checkpoints = @(
    @{ name = "resnet50-0676ba61.pth"; url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"; targetDir = $teachersDir },
    @{ name = "resnet18-f37072fd.pth"; url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"; targetDir = $studentsDir },
    @{ name = "mobilenet_v2-b0353104.pth"; url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"; targetDir = $studentsDir },
    @{ name = "efficientnet_b0_rwightman-3dd342df.pth"; url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"; targetDir = $studentsDir }
)

# create dirs
foreach ($d in @($modelsDir, $teachersDir, $studentsDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# downloader helper
function Download-File($url, $outPath) {
    if (Test-Path $outPath) {
        Write-Host "Already exists: $outPath -- skipping download."
        return
    }
    Write-Host "Downloading $url -> $outPath"
    try {
        # prefer curl if available (Windows 10+)
        if (Get-Command curl -ErrorAction SilentlyContinue) {
            curl -L $url -o $outPath --progress-bar
        } else {
            Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing
        }
    } catch {
        Write-Error "Download failed for $url : $_"
        throw $_
    }
}

# download + compute checksum
$manifest = [System.Collections.Generic.List[object]]::new()
foreach ($ck in $checkpoints) {
    $targetFile = Join-Path $ck.targetDir $ck.name
    Download-File $ck.url $targetFile

    if (-not (Test-Path $targetFile)) {
        throw "File not found after download: $targetFile"
    }

    $hash = Get-FileHash -Path $targetFile -Algorithm SHA256
    $entry = @{
        filename = $ck.name
        url = $ck.url
        path = (Resolve-Path $targetFile).Path
        sha256 = $hash.Hash
        size_bytes = (Get-Item $targetFile).Length
        downloaded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $manifest.Add($entry)
    Write-Host ("{0}  {1} bytes  sha256={2}" -f $ck.name, $entry.size_bytes, $entry.sha256)
}

# write manifest pretty JSON
$manifest | ConvertTo-Json -Depth 5 | Out-File -FilePath $manifestPath -Encoding UTF8
Write-Host "Manifest written to: $manifestPath"
