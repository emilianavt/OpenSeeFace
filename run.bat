%ECHO OFF

facetracker -l 1

echo Make sure that nothing is accessing your camera before you proceed.

set /p cameraNum=Select your camera from the list above and enter the corresponding number:

facetracker -a %cameraNum%

set /p dcaps=Select your camera mode or -1 for default settings:
set /p width=Select the width:
set /p height=Select the height:
set /p fps=Select the FPS:

facetracker -c %cameraNum% -W %width% -H %height% -F %fps% -D %dcaps% -v 3 -P 1 --discard-after 0 --scan-every 0 --no-3d-adapt 1 --max-feature-updates 900

pause