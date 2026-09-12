@echo off
cd /d "%~dp0"
python run_experiments.py all
if errorlevel 1 goto failed
echo Tables D and F completed.
pause
exit /b 0
:failed
echo Experiment failed. Completed simulations are retained for resume.
pause
exit /b 1
