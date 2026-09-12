@echo off
python "%~dp0..\project.py" run --problem Q3 %*
if errorlevel 1 exit /b 1
python "%~dp0..\project.py" package
exit /b %errorlevel%
