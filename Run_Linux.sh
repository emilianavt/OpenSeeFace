#!/bin/bash
# Modern Linux Setup and Launch Script for OpenSeeFace
# Automatically provisions a modern Python environment with CUDA 13 support

# Navigate to the script's directory so it can be run from anywhere
cd "$(dirname "$0")" || exit

echo "Initializing OpenSeeFace Linux Environment..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv env
    source env/bin/activate
    
    echo "Installing modern dependencies (CUDA 13, ONNXRuntime 1.29.0+, Numpy 2+, OpenCV 5+)..."
    pip install --upgrade pip
    pip install onnxruntime-gpu opencv-python pillow numpy nvidia-cublas nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse
    echo "Setup complete!"
else
    source env/bin/activate
fi

# Fix ONNX Runtime missing CUDA 13 libraries by adding pip modules to LD_LIBRARY_PATH dynamically
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cu13/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

echo "Starting OpenSeeFace Tracker on GPU..."
# Execute facetracker
python facetracker.py -c 0 -W 1280 -H 720 --discard-after 0 --scan-every 0 --no-3d-adapt 1 --max-feature-updates 900
