# PowerShell 命令失败时停止脚本。
$ErrorActionPreference = "Stop"
# 计算项目根目录并切换过去，使 Python 使用的相对路径保持一致。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
# 使用项目 .venv 的 Python。
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
# 将 Hugging Face 下载缓存统一放在项目 data/cache 下。
# 这些设置只作用于当前脚本进程及其启动的 Python 子进程。
$env:HF_HOME = Join-Path $ProjectRoot "data\cache\huggingface"
$env:HF_DATASETS_CACHE = Join-Path $ProjectRoot "data\cache\datasets"
$env:HF_HUB_CACHE = Join-Path $ProjectRoot "data\cache\hub"
# 若缓存目录不存在，先创建；Out-Null 隐藏目录对象输出。
New-Item -ItemType Directory -Force -Path $env:HF_HOME,$env:HF_DATASETS_CACHE,$env:HF_HUB_CACHE | Out-Null
# 调用数据准备模块，并原样转发用户传入的可选命令行参数。
& $Python -m src.prepare_data @args
# Python 以非零状态结束时，向调用者明确报错。
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed." }
