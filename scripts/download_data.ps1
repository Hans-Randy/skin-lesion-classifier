# Download HAM10000 from Kaggle into data/.
# Requires: kaggle CLI configured (%USERPROFILE%\.kaggle\kaggle.json with API token).
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path "data" | Out-Null

kaggle datasets download kmader/skin-cancer-mnist-ham10000 -p data/
Expand-Archive -Force -Path "data\skin-cancer-mnist-ham10000.zip" -DestinationPath "data\"

Write-Host "Done. Verify: Get-ChildItem data\HAM10000_metadata.csv"
