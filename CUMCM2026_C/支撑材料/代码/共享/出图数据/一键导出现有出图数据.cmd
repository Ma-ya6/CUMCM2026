@echo off
cd /d "%~dp0"
python export_plot_data.py
if errorlevel 1 goto failed
echo CSV export completed.
pause
exit /b 0
:failed
echo Export failed. See the error above.
pause
exit /b 1
