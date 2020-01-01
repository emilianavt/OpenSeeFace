REM pyinstaller --onefile ^
REM     --add-binary escapi/escapi_x86.dll;. ^
REM     --add-binary escapi/escapi_x64.dll;. ^
REM     facetracker.py

pyinstaller --onedir ^
    --add-binary escapi/escapi_x86.dll;. ^
    --add-binary escapi/escapi_x64.dll;. ^
    facetracker.py