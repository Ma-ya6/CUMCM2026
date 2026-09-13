@echo off
chcp 65001 >nul
cd /d "%~dp0"
python "%~dp0solve_q1_ab.py" %*
if errorlevel 1 exit /b 1
