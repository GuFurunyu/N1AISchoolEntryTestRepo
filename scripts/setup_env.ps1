# 依赖安装中遇到 PowerShell 错误就停止。
$ErrorActionPreference = "Stop"
# 项目根目录是 scripts 的父目录；依赖装进项目的 .venv。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# 本脚本只安装依赖，不创建虚拟环境；必须先有 .venv。
if (-not (Test-Path $Python)) {
    throw "Project Python is missing. Create .venv first."
}

# pip 下载缓存放在项目内，方便查看或清理。
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot "data\cache\pip"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null
# 从 PyTorch 的 CUDA 12.4 软件源安装指定版本的 torch。
& $Python -m pip install --disable-pip-version-check --no-input "torch==2.5.1" --index-url "https://download.pytorch.org/whl/cu124"
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
# 其余 Python 库按 requirements.txt 的锁定版本从 PyPI 安装。
& $Python -m pip install --disable-pip-version-check --no-input -r (Join-Path $ProjectRoot "requirements.txt") --index-url "https://pypi.org/simple"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "Environment setup completed."
