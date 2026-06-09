$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Không tìm thấy .venv\Scripts\python.exe"
    Write-Host "Hãy tạo môi trường trước, hoặc chạy bằng python agent_dashboard.py"
    exit 1
}

Write-Host "Starting Legal Multi-Agent Dashboard..."
Write-Host "Open: http://127.0.0.1:8088"
Write-Host "Press Ctrl+C to stop."
& $Python agent_dashboard.py
