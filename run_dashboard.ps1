# ============================================================
# ชื่อโค้ด: run_dashboard.ps1
# ที่อยู่ไฟล์: C:\Data\Bot\Bot_v2_GroqAI_Analysze_Decision\run_dashboard.ps1
# คำสั่งรัน: powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
# เวอร์ชัน: v1.0.0
# ============================================================

$ErrorActionPreference = "Stop"

Set-Location "C:\Data\Bot\Bot_v2_GroqAI_Analysze_Decision"

Write-Host "====================================="
Write-Host " BOT_v2 PRODUCTION DASHBOARD START"
Write-Host "====================================="

python app/terminal_dashboard.py