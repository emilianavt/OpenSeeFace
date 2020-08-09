REM pyinstaller --onefile ^
REM     --add-binary dshowcapture/dshowcapture_x86.dll;. ^
REM     --add-binary dshowcapture/dshowcapture_x64.dll;. ^
REM     --add-binary escapi/escapi_x86.dll;. ^
REM     --add-binary escapi/escapi_x64.dll;. ^
REM     --add-binary run.bat;. ^
REM     facetracker.py

pyinstaller --onedir ^
    --add-binary dshowcapture/dshowcapture_x86.dll;. ^
    --add-binary dshowcapture/dshowcapture_x64.dll;. ^
    --add-binary escapi/escapi_x86.dll;. ^
    --add-binary escapi/escapi_x64.dll;. ^
    --add-binary run.bat;. ^
    facetracker.py