$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project Python is missing. Create .venv first."
}

$env:PIP_CACHE_DIR = Join-Path $ProjectRoot "data\cache\pip"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null
& $Python -m pip install --disable-pip-version-check --no-input "torch==2.5.1" --index-url "https://download.pytorch.org/whl/cu124"
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
& $Python -m pip install --disable-pip-version-check --no-input -r (Join-Path $ProjectRoot "requirements.txt") --index-url "https://pypi.org/simple"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "Environment setup completed."
