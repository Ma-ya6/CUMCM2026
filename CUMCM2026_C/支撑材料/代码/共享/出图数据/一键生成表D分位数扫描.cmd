@echo off
cd /d "%~dp0"
python run_experiments.py D
if errorlevel 1 goto failed
echo Table D completed.
pause
exit /b 0
:failed
echo Experiment failed. Completed points are retained for resume.
pause
exit /b 1
