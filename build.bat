@echo off
echo ============================================================
echo   NeuralScribe v2 — Build Script
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

:: Install build tools
echo [1/4] Installing build tools...
pip install nuitka ordered-set zstandard --quiet
pip install -r requirements.txt --quiet
echo Done.
echo.

:: Create output directories
if exist "dist" rmdir /s /q "dist"
mkdir dist

:: Compile with Nuitka
echo [2/4] Compiling with Nuitka (this takes 10-30 minutes)...
echo.
python -m nuitka ^
    --standalone ^
    --output-dir=dist ^
    --output-filename=NeuralScribe.exe ^
    --windows-console-mode=attach ^
    --windows-icon-from-ico=icon.ico ^
    --enable-plugin=torch ^
    --include-data-dir=frontend=frontend ^
    --include-data-dir=configs=configs ^
    --include-data-files=project_config.json=project_config.json ^
    --include-data-files=requirements.txt=requirements.txt ^
    --include-module=uvicorn ^
    --include-module=uvicorn.logging ^
    --include-module=uvicorn.protocols ^
    --include-module=uvicorn.protocols.http ^
    --include-module=uvicorn.protocols.http.httptools_impl ^
    --include-module=uvicorn.protocols.http.h11_impl ^
    --include-module=uvicorn.protocols.websockets ^
    --include-module=uvicorn.protocols.websockets.websockets_impl ^
    --include-module=uvicorn.lifespan ^
    --include-module=uvicorn.lifespan.on ^
    --include-module=fastapi ^
    --include-module=pydantic ^
    --include-module=eel ^
    --include-module=pynvml ^
    --include-module=psutil ^
    --include-module=yaml ^
    --include-module=PIL ^
    --include-module=scipy ^
    --include-module=matplotlib ^
    --include-module=torchvision ^
    --include-module=onnx ^
    --include-module=onnxruntime ^
    launcher.py

if errorlevel 1 (
    echo.
    echo ERROR: Nuitka compilation failed.
    pause
    exit /b 1
)

:: Create empty data directories in dist
echo [3/4] Creating data directories...
set DIST=dist\launcher.dist
mkdir "%DIST%\datasets\english\cache" 2>nul
mkdir "%DIST%\datasets\english\raw" 2>nul
mkdir "%DIST%\models\english\exports" 2>nul
mkdir "%DIST%\training_logs\english" 2>nul
echo [] > "%DIST%\models\english\index.json"
echo Done.
echo.

echo [4/4] Build complete!
echo.
echo   Output: dist\launcher.dist\NeuralScribe.exe
echo.
echo   To create an installer, install Inno Setup and compile installer.iss
echo.
pause
