@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

echo ============================================================
echo  创建虚拟环境 .venv 并安装 requirements.txt
echo  项目根：%CD%
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.10+ 并加入 PATH。
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo [跳过] .venv 已存在，仅更新依赖。
) else (
    echo [1/2] 创建虚拟环境 .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
)

echo [2/2] 安装依赖 ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络或 pip 源。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  完成。后续运行请使用：
echo      .venv\Scripts\python.exe 代码\共享\project.py run
echo ============================================================
pause
