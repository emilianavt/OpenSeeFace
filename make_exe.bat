REM pyinstaller facetracker.spec --onefile ^
REM     --add-binary dshowcapture/dshowcapture_x86.dll;. ^
REM     --add-binary dshowcapture/dshowcapture_x64.dll;. ^
REM     --add-binary escapi/escapi_x86.dll;. ^
REM     --add-binary escapi/escapi_x64.dll;. ^
REM     --add-binary run.bat;.

pyinstaller facetracker.spec --onedir ^
    --add-binary dshowcapture/dshowcapture_x86.dll;. ^
    --add-binary dshowcapture/dshowcapture_x64.dll;. ^
    --add-binary escapi/escapi_x86.dll;. ^
    --add-binary escapi/escapi_x64.dll;. ^
    --add-binary run.bat;.

del dist\facetracker\cv2\opencv_videoio_ffmpeg420_64.dll

pause