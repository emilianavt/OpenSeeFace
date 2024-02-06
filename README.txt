-----How to Install-----
This is for linux, idk how python packages work in windows
*Download the project, there's not really a "release" because it's a bunch of .py files, there's nothing to compile or build
*Open a terminal in the project folder
*Create a virtual environment using 'python3 -m venv .venv'
*Activate the virtual environment with 'source .venv/bin/activate'
*Using pip, install the following packages:
    numpy
    onnxruntime
    opencv-python

On the Vtube Studio side:
*go to .../VTube Studio/VTube Studio_Data/StreamingAssets/
*create a file named 'ip.txt'
*put the following in ip.txt:
# To listen for remote connections, change this to 0.0.0.0 or your actual IP on the desired interface.
ip=127.0.0.1

# This is the port the server will listen for tracking packets on.
port=11573

And you're done, it's installed
you can run it by running my facetracking.sh shell script which applies some settings I found useful for my camera (though I'm unsure how compatible they are with other setups), or you can just use 'python3 facetracker.py' after re-activating the virtual environment

