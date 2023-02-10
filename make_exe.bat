echo "Started"

:: A venv is used so it's easy to add the onnxruntime dll to the binary

echo "Creating venv"
python -m venv venv

echo "Activating venv"
call venv\Scripts\activate.bat

echo "Installing dependencies"
pip install wheel

pip install onnxruntime opencv-python==4.5.4.60 pillow numpy==1.23.0 pyinstaller

echo "Running pyinstaller"
pyinstaller facetracker.py --clean ^
    --onedir ^
    --add-binary dshowcapture/*.dll;. ^
    --add-binary escapi/*.dll;. ^
    --add-binary venv/lib/site-packages/onnxruntime/capi/*.dll;onnxruntime\capi ^
    --add-binary msvcp140.dll;. ^
    --add-binary vcomp140.dll;. ^
    --add-binary concrt140.dll;. ^
    --add-binary vccorlib140.dll;. ^
    --add-binary run.bat;.

echo "Deleting opencv dll"
del dist\facetracker\cv2\opencv_videoio_*

echo "Finished"
