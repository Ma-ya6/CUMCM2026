@echo off
chcp 65001 >nul
setlocal
rem 共享 目录名含中文，直接写字面量在部分代码页下解析失败，
rem 因此用 for /d 让 cmd 自己展开目录名，再按 project.py 定位共享内核。
set "SHARED="
for /d %%D in ("%~dp0..\..\*") do if exist "%%D\project.py" if not defined SHARED set "SHARED=%%D"
if not defined SHARED (
    echo [错误] 未找到 共享\project.py，请保持 代码\ 目录结构完整。
    exit /b 1
)
python "%SHARED%\project.py" run --problem Q4-2 %*
if errorlevel 1 exit /b 1
