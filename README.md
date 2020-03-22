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

The included `OpenSeeLauncher` component allows starting the face tracker program from Unity. It is designed to work with the pyinstaller created executable distributed in the binary release bundles. It provides three public API functions:

* `public string[] ListCameras()` returns the names of available cameras. The index of the camera in the array corresponds to its ID for the `cameraIndex` field. Setting the `cameraIndex` to `-1` will disable webcam capturing.
* `public bool StartTracker()` will start the tracker. If it is already running, it will shut down the running instance and start a new one with the current settings.
* `public void StopTracker()` will stop the tracker. The tracker is stopped automatically when the application is terminated or the `OpenSeeLauncher` object is destroyed.

The `OpenSeeLauncher` component uses WinAPI job objects to ensure that the tracker child process is terminated if the application crashes.

Additional custom commandline arguments should be added one by one into elements of `commandlineArguments` array. For example `-v 1` should be added as two elements, one element containing `-v` and one containing `1`, not a single one containing both parts.

## Expression detection

The `OpenSeeExpression` component can be added to the same component as the `OpenSeeFace` component to detect specific facial expressions. It has to be calibrated on a per-user basis. It can be controlled either through the checkboxes in the Unity Editor or through the equivalent public methods that can be found in its source code.

To calibrate this system, you have to gather example data for each expression. The way `OpenSeeExpression` is set up, it requires 400 examples of each expression, which should take around 30 seconds to capture. If capture is too fast, you can use the `recordingSkip` option to slow it down.

The general process is as follows:

* Type in a name for the expression you want to calibrate.
* Make the expression and hold it, then tick the recording box.
* Keep holding the expression and move your head around and turn it in various directions.
* After a short while, start talking while doing so if the expression should be compatible with talking.
* When the "Percent Recorded" field is at around 50, untick the recording box and work on capturing another expression.
* When you have them all at around 50, go back to the first and go through them again to bring all the way to 100%.
* Then tick the train box and you should get some statistics in the lower part.
* Tick the predict box and it will show you which expression you are making.

To delete the captured data for an expression, type in its name and tick the "Clear" box. To save both the trained model and the captured training data, type in a filename including its full path in the "Filename" field and tick the "Save" box. To load it, enter the filename and tick the "Load" box.

Up to 25 expressions are supported, but a more reasonable number is 5-6.

# General notes

* The tracking seems to be quite robust even with partial occlusion of the face, glasses or bad lighting conditions.
* There is now an experimental gaze tracking model.
* Depending on the frame rate, face tracking can easily use up a whole CPU core. At 30fps for a single face, it should still use less than 100% of one core on a decent CPU.
* When setting the number of faces to track to a higher number than the number of faces actually in view, the OpenCV face detection will attempt to find new faces every `--scan-every` frames. It can be quite slow, so try to set `--faces` no higher than the actual number of faces you are tracking.

# Models

Four pretrained models are included. Using the `--model` switch, it is possible to select them for tracking.

* Model **0**: This is a very fast, low accuracy model based on ShuffleNetV2. (67fps)
* Model **1**: This is a slightly slower model with better accuracy based on MobileNetV3. (62fps)
* Model **2**: This is a slower model with good accuracy based on ShuffleNetV2. (45fps)
* Model **3** (default): This is the slowest and highest accuracy model. It is based on MobileNetV3. (42fps)

FPS measurements are from running on one core of my CPU.

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

For the training the gaze and blink detection model, the [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) dataset was used. Additionally, around 125000 synthetic eyes generated with [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) were used during training.

## Algorithm

The algorithm is inspired by:

* [Designing Neural Network Architectures for Different Applications: From Facial Landmark Tracking to Lane Departure Warning System](https://www.synopsys.com/designware-ip/technical-bulletin/ulsee-designing-neural-network.html) by YiTa Wu, Vice President of Engineering, ULSee
* [Real-time Human Pose Estimation in the Browser with TensorFlow.js](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) by Olaf Ronneberger, Philipp Fischer, Thomas Brox
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
* [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) by Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) by Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam

The ShuffleNet V2 code is taken from `torchvision`. The MobileNetV3 code was taken from [here](https://github.com/rwightman/gen-efficientnet-pytorch).

For all training after the first model, a modified version of [Adaptive Wing Loss](https://github.com/tankrant/Adaptive-Wing-Loss) was used.

* [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399) by Xinyao Wang, Liefeng Bo, Li Fuxin

For expression detection, [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is used.

Face detection is done using a custom heatmap regression based face detection model or RetinaFace.

    @inproceedings{deng2019retinaface,
      title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
      author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
      booktitle={arxiv},
      year={2019}
    }

RetinaFace detection is based on [this](https://github.com/biubug6/Pytorch_Retinaface) implementation. The pretrained model was modified to remove unnecessary landmark detection and converted to ONNX format for a resolution of 640x640.

# License

The code and models are distributed under the BSD 2-clause license. 

`models/haarcascade_frontalface_alt2.xml` is distributed under its own license. You can find licenses of third party libraries used for binary builds in the `Licenses` folder.

