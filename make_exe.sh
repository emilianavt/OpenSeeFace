#!/bin/bash

PYTHON_BINARY=$(which python3)

if [[ -z "$PYTHON_BINARY" ]]; then
    echo "python3 not found. Searching for python instead."
    PYTHON_BINARY=$(which python)
    if [[ -z "$PYTHON_BINARY" ]]; then
        echo "Python 3 must be available on your path. Exiting."
        exit 1
    fi
fi

echo "Creating venv"

"$PYTHON_BINARY" -m venv venv

source venv/bin/activate

echo "Installing packages"

pip install onnxruntime opencv-python==4.5.4.60 pillow numpy pyinstaller

echo "Creating binary"

pyinstaller --clean facetracker.py

echo "Files should be available in the dist/ folder"
echo "Done!"

