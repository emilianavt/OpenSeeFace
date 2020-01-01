# Overview

This project implements a facial landmark detection model based on ShuffleNetV2.

As Pytorch 1.3 CPU inference speed on Windows is very low, the model was converted to ONNX format. Using [onnxruntime](https://github.com/microsoft/onnxruntime) it can run at 30 - 60 fps tracking a single face. A third and smaller model runs at over 60 fps tracking a single face on a single CPU core.

If anyone is curious, the name is a silly pun on the open seas and seeing faces. There's no deeper meaning.

Thanks to [@Virtual_Deat](https://twitter.com/Virtual_Deat) for helping me test everything.

[Unity sample video](https://twitter.com/emiliana_vt/status/1210622149314203648) | [Sample video](https://www.youtube.com/watch?v=AOPHiAp9DBE) | [Sample video](https://www.youtube.com/watch?v=-cBSuHGdBWQ)

# Usage

The face tracking itself is done by the `facetracker.py` Python 3.7 script. It is a commandline program, so you should start it manually from cmd or write a batch file to start it.

The script will perform the tracking on a webcam or video file and send the tracking data over UDP. This design also allows tracking to be done on a separate PC from the one who uses the tracking information. This can be useful to enhance performance and to avoid accidentially revealing camera footage.

The provided `OpenSee` Unity component can receive these UDP packets and provides the received information through a public field called `trackingData`. The `OpenSeeShowPoints` component can visualize the landmark points of the first detected face. It also serves as an example. Please look at it to see how to properly make use of the `OpenSee` component. The UDP packets are received in a separate thread, so any components using the `trackingData` field of the `OpenSee` component should first copy the field and access this copy, because otherwise the information may get overwritten during processing.

Run the python script with `--help` to learn about the possible options you can set.

    python facetracker.py --help

A simple demonstration can be achieved by creating a new scene in Unity, adding an empty game object and both the `OpenSee` and `OpenSeeShowPoints` components to it. While the scene is playing, run the face tracker on a video file:

    python facetracker.py --visualize 1 --max-threads 2 -c video.mp4

This way the tracking script will output its own tracking visualization while also demonstrating the transmission of tracking data to Unity.

The included `OpenSeeLauncher` component allows starting the face tracker program from Unity. It is designed to work with the pyinstaller created executable distributed in the binary release bundles.

# General notes

* The tracking seems to be quite robust even with partial occlusion of the face, glasses or bad lighting conditions.
* There is no gaze direction tracking, and the model seems prone to thinking eyes are opened, even when they are closed.
* Depending on the frame rate, face tracking can easily use up a whole CPU core. At 30fps for a single face, it should still use less than 100% of one core on a decent CPU.
* When setting the number of faces to track to a higher number than the number of faces actually in view, the OpenCV face detecter will attempt to find new faces every `--scan-every` frames. It can be quite slow, so try to set `--faces` no higher than the actual number of faces you are tracking.

# Models

Three pretrained models are included. Using the `--model` switch, it is possible to select them for tracking.

* Model **0**: This is the original model I trained. It is the most reliable for general 3D pose tracking, but it has difficulties with eyebrow and eye positions.
* Model **1** (default): This model trades off a little bit of robustness for better facial feature tracking.
* Model **2**: This is a smaller model that can run much faster, easily reaching 60 - 90 fps on a single CPU core. However, the tracking robustness and accuracy are lower.

# Release builds

The builds in the release section of this repository contain a `facetracker.exe` inside a `Binary` folder that was built using `pyinstaller` and contains all required dependencies.

To run it, at least the `models` and `escapi` folders have to be placed in the same folder as `facetracker.exe`.

When distributing it, you should also distribute the `Licenses` folder along with it to make sure you conform to requirements set forth by some of the third party libraries. Unused models can be removed from redistributed packages without issue.

# Dependencies

* Python 3.7
* ONNX Runtime
* OpenCV
* Pillow
* Numpy

The required libraries can be installed using pip:

     pip install onnxruntime opencv-python pillow numpy

# References

## Training dataset

The model was trained on a 66 point version of the [LS3D-W](https://www.adrianbulat.com/face-alignment) dataset.

    @inproceedings{bulat2017far,
      title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
      author={Bulat, Adrian and Tzimiropoulos, Georgios},
      booktitle={International Conference on Computer Vision},
      year={2017}
    }

Additional training has been done on the WFLW dataset after reducing it to 66 points and replacing the contour points and tip of the nose with points predicted by the model trained up to this point. This additional training is done to improve fitting to eyes and eyebrows.

    @inproceedings{wayne2018lab,
      author = {Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
      title = {Look at Boundary: A Boundary-Aware Face Alignment Algorithm},
      booktitle = {CVPR},
      month = June,
      year = {2018}
    }

The original model can be selected with `--model 0` instead.

## Algorithm

The algorithm is inspired by:

* [Designing Neural Network Architectures for Different Applications: From Facial Landmark Tracking to Lane Departure Warning System](https://www.synopsys.com/designware-ip/technical-bulletin/ulsee-designing-neural-network.html) by YiTa Wu, Vice President of Engineering, ULSee
* [Real-time Human Pose Estimation in the Browser with TensorFlow.js](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) by Olaf Ronneberger, Philipp Fischer, Thomas Brox
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
* [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) by Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

The ShuffleNet V2 code is taken from `torchvision`.

For all training after the first model, a modified version of [Adaptive Wing Loss](https://github.com/tankrant/Adaptive-Wing-Loss) was used.

* [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399) by Xinyao Wang, Liefeng Bo, Li Fuxin

# License

The code and model are distributed under the BSD 2-clause license. 

`models/haarcascade_frontalface_alt2.xml` is distributed under its own license. You can find licenses of third party libraries used for binary builds in the `Licenses` folder.

