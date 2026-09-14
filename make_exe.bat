echo "Started"

:: A venv is used so it's easy to add the onnxruntime dll to the binary

echo "Installing uv"
pip install uv

echo "Installing dependencies"
uv sync --locked
uv pip install pyinstaller==6.22.3

echo "Running pyinstaller"
uv run pyinstaller facetracker.py --clean ^
    --onedir ^
    --add-binary dshowcapture/*.dll;. ^
    --add-binary escapi/*.dll;. ^
    --add-binary .venv/lib/site-packages/onnxruntime/capi/*.dll;onnxruntime\capi ^
    --add-binary msvcp140.dll;. ^
    --add-binary vcomp140.dll;. ^
    --add-binary concrt140.dll;. ^
    --add-binary vccorlib140.dll;. ^
    --add-binary run.bat;.

echo "Deleting opencv dll"
del /f dist\facetracker\_internal\cv2\opencv_videoio_*

echo "Finished"

