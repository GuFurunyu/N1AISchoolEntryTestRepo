$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"

# 解压到新位置后，随项目携带的 Windows venv 可能仍指向发送者的旧路径。
# 训练入口在这种情况下先调用环境准备脚本，再继续执行训练。
$NeedsSetup = -not (Test-Path $Python)
if (-not $NeedsSetup) {
    $BasePrefix = & $Python -c "import sys; print(sys.base_prefix)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $BasePrefix "python.exe"))) {
        $NeedsSetup = $true
    }
}
if ($NeedsSetup) {
    & (Join-Path $PSScriptRoot "setup_env.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Project environment setup failed." }
}

& $Python -m src.train --config configs/english_single/base.yaml --resume auto @args
if ($LASTEXITCODE -ne 0) { throw "English single-language training stopped with an error." }
