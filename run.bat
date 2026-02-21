@echo off
setlocal EnableExtensions

pushd "%~dp0" >nul

set "TARGET_IP=127.0.0.1"
set "TARGET_PORT=11573"
set "TRACKER_EXE=%~dp0facetracker.exe"
set "TRACKER_EXE_BINARY=%~dp0Binary\facetracker.exe"
set "TRACKER_PY=%~dp0facetracker.py"
set "TRACKER_MODE="

if exist "%TRACKER_EXE%" (
    set "TRACKER_MODE=exe"
) else if exist "%TRACKER_EXE_BINARY%" (
    set "TRACKER_MODE=exe"
    set "TRACKER_EXE=%TRACKER_EXE_BINARY%"
) else if exist "%TRACKER_PY%" (
    set "TRACKER_MODE=python"
) else (
    echo Error: Could not find facetracker.exe, Binary\facetracker.exe, or facetracker.py.
    goto :finish
)

if /I "%TRACKER_MODE%"=="python" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Error: Python is required to run facetracker.py, but python was not found in PATH.
        goto :finish
    )
    python -c "import onnxruntime, cv2, PIL, numpy" >nul 2>&1
    if errorlevel 1 (
        echo Error: Missing Python dependencies for facetracker.py.
        echo Install with: pip install onnxruntime opencv-python pillow numpy
        goto :finish
    )
)

echo Streaming target: %TARGET_IP%:%TARGET_PORT%
call :run_tracker -l 1
if errorlevel 1 goto :finish

echo Make sure that nothing is accessing your camera before you proceed.
set /p "cameraNum=Select your camera from the list above and enter the corresponding number: "

call :run_tracker -a %cameraNum%
if errorlevel 1 goto :finish

set /p "dcaps=Select your camera mode or -1 for default settings [default: -1]: "
if not defined dcaps set "dcaps=-1"
set /p "fps=Select the FPS [default: 24]: "
if not defined fps set "fps=24"

call :run_tracker -c %cameraNum% -F %fps% -D %dcaps% -v 3 -P 1 --discard-after 0 --scan-every 0 --no-3d-adapt 1 --max-feature-updates 900 -i %TARGET_IP% -p %TARGET_PORT%

:finish
popd >nul
pause
exit /b

:run_tracker
if /I "%TRACKER_MODE%"=="exe" (
    "%TRACKER_EXE%" %*
) else (
    python "%TRACKER_PY%" %*
)
exit /b %errorlevel%
