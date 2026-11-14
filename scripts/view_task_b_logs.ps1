param([int]$Port = 6007)

# 从项目根目录读取当前英文单语 Task B 四条路径的 TensorBoard 事件文件。
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m tensorboard.main --logdir "experiments/english_single/task_b/tensorboard" --port $Port
