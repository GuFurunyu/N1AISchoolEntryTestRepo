# PowerShell 自身报错时停止。
$ErrorActionPreference = "Stop"
# 切换到项目根目录，使 runs 相对于项目解析。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
# 调用项目虚拟环境中的 TensorBoard。
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
# 启动日志网页服务；运行时保持这个窗口打开，在浏览器查看 localhost:6006。
& $Python -m tensorboard.main --logdir runs --port 6006
