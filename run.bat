%ECHO OFF

facetracker -l 1


set /p cameraNum=Select your camera from the list above and enter the corresponding number:
set /p width=Select the width:
set /p height=Select the height:
set /p fps=Select the FPS:

facetracker -c %cameraNum% -W %width% -H %height% -F %fps% -v 1 -P 1

pause