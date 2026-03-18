@echo off
echo ========================================================
echo OracleBot-Pro: Starting Production Mode (Dual Process)
echo ========================================================
echo.

:: กำหนดค่า Environment Variables สำหรับรันบอทในโหมด Production จริง
set RUN_PRODUCTION_RUNTIME=1
set RUN_POSITION_MONITOR=1
set ENABLE_ENTRY_EXECUTION=1
set ENABLE_POSITION_CLOSE_EXECUTION=1
set PRODUCTION_ENTRY_MAGIC=190058
set PRODUCTION_ENTRY_COMMENT=Groq_AI

:: สร้างโฟลเดอร์สำหรับเก็บ Log ของการทำงาน
if not exist "storage\logs" mkdir "storage\logs"

echo [1/2] Starting Position Monitor (Exit Management)...
:: ใช้คำสั่ง start เพื่อรันสคริปต์นี้ในหน้าต่างใหม่ (หรือ background)
start "Bot Position Monitor" cmd /c "set RUN_PRODUCTION_RUNTIME=0 && set RUN_POSITION_MONITOR=1 && python app/main.py 2>&1"

echo [2/2] Starting Production Runtime (Entry Detection & AI)...
:: รันสคริปต์หลักในหน้าต่างปัจจุบัน
set RUN_PRODUCTION_RUNTIME=1
set RUN_POSITION_MONITOR=0
python app/main.py

echo.
echo Production Runtime stopped.
pause
