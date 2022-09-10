pip install onnxruntime opencv-python==4.5.4.60 pillow numpy pyinstaller

pyinstaller facetracker.py --onedir ^
    --add-binary dshowcapture/dshowcapture_x86.dll;. ^
    --add-binary dshowcapture/dshowcapture_x64.dll;. ^
    --add-binary dshowcapture/libminibmcapture32.dll;. ^
    --add-binary dshowcapture/libminibmcapture64.dll;. ^
    --add-binary escapi/escapi_x86.dll;. ^
    --add-binary escapi/escapi_x64.dll;. ^
    --add-binary msvcp140.dll;. ^
    --add-binary vcomp140.dll;. ^
    --add-binary concrt140.dll;. ^
    --add-binary vccorlib140.dll;. ^
    --add-binary run.bat;.

del dist\facetracker\cv2\opencv_videoio_*