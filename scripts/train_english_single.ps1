$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
& $Python -m src.train --config configs/english_single/base.yaml --resume auto @args
if ($LASTEXITCODE -ne 0) { throw "English single-language training stopped with an error." }
