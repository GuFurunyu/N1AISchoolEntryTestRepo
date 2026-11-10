$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
& $Python -m src.train --config configs/base.yaml --resume auto @args
if ($LASTEXITCODE -ne 0) { throw "Training stopped with an error." }
