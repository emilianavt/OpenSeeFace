#!/bin/bash

set -e

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
pip install wheel # Make sure this is installed beforehand

pip install onnxruntime opencv-python==4.5.4.60 pillow numpy==1.23.0 pyinstaller

echo "Creating binary"
pyinstaller --onedir --clean facetracker.py \
    --add-binary venv/lib/python3.*/site-packages/onnxruntime/capi/*.so:onnxruntime/capi

echo "Files should be available in the dist/ folder"
echo "Done!"

