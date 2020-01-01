"""
A simple python wrapper around escapi

Usage:

from camera import Device

device = Deveice.connect(0, 500, 500)
image = device.get_image()
"""

import os
import platform
import sys
from ctypes import *
from PIL import Image
import numpy as np
import cv2

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f


class CAPTURE_PROPETIES:
    CAPTURE_BRIGHTNESS = 1,
    CAPTURE_CONTRAST = 2,
    CAPTURE_HUE = 3,
    CAPTURE_SATURATION = 4,
    CAPTURE_SHARPNESS = 5,
    CAPTURE_GAMMA = 6,
    CAPTURE_COLORENABLE = 7,
    CAPTURE_WHITEBALANCE = 8,
    CAPTURE_BACKLIGHTCOMPENSATION = 9,
    CAPTURE_GAIN = 10,
    CAPTURE_PAN = 11,
    CAPTURE_TILT = 12,
    CAPTURE_ROLL = 13,
    CAPTURE_ZOOM = 14,
    CAPTURE_EXPOSURE = 15,
    CAPTURE_IRIS = 16,
    CAPTURE_FOCUS = 17,
    CAPTURE_PROP_MAX = 18,


class SimpleCapParms(Structure):
    _fields_ = [
        ("buffer", POINTER(c_int)),
        ("width", c_int),
        ("height", c_int),
        ("fps", c_int),
    ]


lib = None


def init():
    global lib
    if platform.architecture()[0] == '32bit':
        dll_path = resolve(os.path.join("escapi", "escapi_x86.dll"))
        lib = cdll.LoadLibrary(dll_path)
    else:
        dll_path = resolve(os.path.join("escapi", "escapi_x64.dll"))
        lib = cdll.LoadLibrary(dll_path)
    if lib is None or lib.ESCAPIVersion() != 0xfff001:
        print("Invalid ESCAPI DLL found.")
        sys.exit(1)
    lib.initCapture.argtypes = [c_int, POINTER(SimpleCapParms)]
    lib.initCapture.restype = c_int
    lib.initCOM()

def count_capture_devices():
    return lib.countCaptureDevices()

def device_name(device):
    """
    Get the device name for the given device
    :param device: The number of the device
    :return: The name of the device
    """
    namearry = (c_char * 256)()
    lib.getCaptureDeviceName(device, namearry, 256)
    camearaname = namearry.value
    return camearaname

def init_camera(device, width, height, fps):
    global devices
    array = (width * height * c_int)()
    options = SimpleCapParms()
    options.width = width
    options.height = height
    options.fps = fps
    options.buffer = array
    lib.initCapture(device, byref(options))
    return array

def do_capture(device):
    lib.doCapture(device)

def is_capture_done(device):
    return lib.isCaptureDone(device)

def read(device, width, height, array):
    if is_capture_done(device):
        img = Image.frombuffer('RGBA', (width, height), array, 'raw', 'BGRA', 0, 1)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR);
        return img
    else:
        return None

def get_image(device, width, height, array):
    lib.doCapture(device)

    while lib.isCaptureDone(device) == 0:
        pass

    img = Image.frombuffer('RGBA', (width, height), array, 'raw', 'BGRA', 0, 0)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
    return img

def deinit_camera(device):
    lib.deinitCapture(device)
