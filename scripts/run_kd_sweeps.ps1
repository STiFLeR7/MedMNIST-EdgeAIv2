param(
  [string]$Datasets = "ham10000,medmnist,isic",
  [string]$Students = "resnet18,mbv2,effb0",
  [string]$Device   = "cuda",
  [int]$Epochs      = 6,
  [string]$OutRoot  = ".\reports"
)

# Lightweight grid: just enough to populate heatmaps/trends on 3050
$alphas = @("0.2","0.5","0.8")
$taus   = @("2","4","8")
$betas  = @("0.0","0.5")   # AT on/off

$PY = "python"
$dsList = $Datasets.Split(",")
$stList = $Students.Split(",")

foreach ($ds in $dsList) {
  foreach ($st in $stList) {
    foreach ($a in $alphas) {
      foreach ($t in $taus) {
        foreach ($b in $betas) {
          $tag = "KD_a${a}_t${t}_b${b}_${st}_${ds}"
          Write-Host "[launch] $ds $st alpha=$a tau=$t beta=$b -> $tag"
          & $PY ".\tools\train_kd.py" `
            --dataset $ds `
            --teacher resnet50 `
            --student $st `
            --alpha $a `
            --tau $t `
            --beta $b `
            --epochs $Epochs `
            --batch 64 `
            --device $Device `
            --out-root $OutRoot `
            --save-tag $tag
        }
      }
    }
  }
}

Write-Host "[OK] KD sweep jobs completed."
