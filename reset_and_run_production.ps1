# ============================================================
# ชื่อโค้ด: reset_and_run_production.ps1
# ที่อยู่ไฟล์: C:\Data\Bot\Bot_v2_GroqAI_Analysze_Decision\reset_and_run_production.ps1
# คำสั่งรัน: powershell -ExecutionPolicy Bypass -File .\reset_and_run_production.ps1
# เวอร์ชัน: v1.0.0
# ============================================================

$ErrorActionPreference = "Stop"

Set-Location "C:\Data\Bot\Bot_v2_GroqAI_Analysze_Decision"

Write-Host "====================================="
Write-Host " RESET + RUN PRODUCTION"
Write-Host "====================================="

Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Remove-Item ".\runtime\dashboard_state.json" -ErrorAction SilentlyContinue
Remove-Item ".\storage\processed_setups.json" -ErrorAction SilentlyContinue

Remove-Item Env:RUN_POSITION_MONITOR -ErrorAction SilentlyContinue
$env:RUN_PRODUCTION_RUNTIME = "1"

python app/main.py