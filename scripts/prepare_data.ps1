$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:HF_HOME = Join-Path $ProjectRoot "data\cache\huggingface"
$env:HF_DATASETS_CACHE = Join-Path $ProjectRoot "data\cache\datasets"
$env:HF_HUB_CACHE = Join-Path $ProjectRoot "data\cache\hub"
New-Item -ItemType Directory -Force -Path $env:HF_HOME,$env:HF_DATASETS_CACHE,$env:HF_HUB_CACHE | Out-Null
& $Python -m src.prepare_data @args
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed." }
