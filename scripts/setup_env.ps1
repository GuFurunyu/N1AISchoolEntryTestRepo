# 依赖安装中遇到 PowerShell 错误就停止。
param(
    # 解压后的 .venv 可能仍记录发送者的旧路径；需要时强制重建它。
    [switch]$RecreateVenv
)
$ErrorActionPreference = "Stop"

# 项目根目录是 scripts 的父目录；依赖装进项目的 .venv。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$BundledPython = Join-Path $ProjectRoot ".runtime\python\python.exe"

function New-ProjectVenv {
    if (Test-Path $BundledPython) {
        $BasePython = $BundledPython
    } else {
        $Command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $Command) {
            throw "No Python interpreter found. Install Python 3.12 or provide .runtime\python\python.exe."
        }
        $BasePython = $Command.Source
    }
    & $BasePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Project virtual environment creation failed." }
}

if ($RecreateVenv -and (Test-Path $VenvDir)) {
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

# Windows venv 会把创建时的基础 Python 路径写入 pyvenv.cfg。
# 项目被移动后，如果旧基础路径不存在，就自动重建本项目环境。
$VenvUsable = Test-Path $Python
if ($VenvUsable) {
    $BasePrefix = & $Python -c "import sys; print(sys.base_prefix)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $BasePrefix "python.exe"))) {
        $VenvUsable = $false
    }
}
if (-not $VenvUsable) {
    if (Test-Path $VenvDir) {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
    New-ProjectVenv
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
