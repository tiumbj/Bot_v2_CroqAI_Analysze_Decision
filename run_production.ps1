# ============================================================
# ชื่อโค้ด: run_production.ps1
# ที่อยู่ไฟล์: C:\Data\Bot\Bot_v2_GroqAI_Analysze_Decision\run_production.ps1
# คำสั่งรัน: powershell -ExecutionPolicy Bypass -File .\run_production.ps1
# เวอร์ชัน: v1.0.0
# ============================================================

$ErrorActionPreference = "Stop"

Set-Location "C:\Data\Bot\Bot_v2_GroqAI_Analysze_Decision"

Write-Host "====================================="
Write-Host " BOT_v2 PRODUCTION RUNTIME START"
Write-Host "====================================="

Remove-Item Env:RUN_POSITION_MONITOR -ErrorAction SilentlyContinue
$env:RUN_PRODUCTION_RUNTIME = "1"

python app/main.py