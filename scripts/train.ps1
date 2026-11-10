# 遇到 PowerShell 错误立即停止，避免后续命令在错误状态下继续运行。
$ErrorActionPreference = "Stop"
# $PSScriptRoot 是本脚本所在的 scripts 目录；上一级是项目根目录。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
# 让配置文件中的相对路径都从项目根目录解析。
Set-Location $ProjectRoot
# 使用项目自带的虚拟环境，不依赖系统默认 python。
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
# 强制 Python 的标准输入输出使用 UTF-8，减少中文路径和文本乱码。
$env:PYTHONUTF8 = "1"
# @args 原样转发调用者额外输入的参数，例如 --max-tokens。
# --resume auto 会尝试从 checkpoints 中最新步数的检查点续训。
& $Python -m src.train --config configs/base.yaml --resume auto @args
# 外部 Python 进程的非零退出码表示执行失败。
if ($LASTEXITCODE -ne 0) { throw "Training stopped with an error." }
