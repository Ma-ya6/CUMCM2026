@echo off
python "%~dp0project.py" package
exit /b %errorlevel%
