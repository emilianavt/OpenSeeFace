import os
import platform
import sys
import numpy as np
from ctypes import *
from PIL import Image
import cv2
import json

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f

lib = None

def create_frame_buffer(width, height):
    buffer = bytearray(width * height * 4 * 4)
    char_array = c_char * len(buffer)
    return char_array.from_buffer(buffer)

class DShowCapture():
    def __init__(self):
        global lib
        if lib is None:
            if platform.architecture()[0] == '32bit':
                dll_path = resolve(os.path.join("dshowcapture", "dshowcapture_x86.dll"))
                lib = cdll.LoadLibrary(dll_path)
            else:
                dll_path = resolve(os.path.join("dshowcapture", "dshowcapture_x64.dll"))
                lib = cdll.LoadLibrary(dll_path)
            lib.create_capture.restype = c_void_p
            lib.get_devices.argtypes = [c_void_p]
            lib.get_device.argtypes = [c_void_p, c_int, c_char_p, c_int]
            lib.capture_device.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
            lib.capture_device_by_dcap.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_longlong]
            lib.capture_device_default.argtypes = [c_void_p, c_int]
            lib.get_width.argtypes = [c_void_p]
            lib.get_height.argtypes = [c_void_p]
            lib.get_fps.argtypes = [c_void_p]
            lib.get_flipped.argtypes = [c_void_p]
            lib.get_colorspace.argtypes = [c_void_p]
            lib.get_colorspace_internal.argtypes = [c_void_p]
            lib.capturing.argtypes = [c_void_p]
            lib.get_frame.argtypes = [c_void_p, c_int, c_char_p, c_int]
            lib.get_size.argtypes = [c_void_p]
            lib.stop_capture.argtypes = [c_void_p]
            lib.get_json_length.argtypes = [c_void_p]
            lib.get_json.argtypes = [c_void_p, c_char_p, c_int]
            lib.destroy_capture.argtypes = [c_void_p]
        self.lib = lib
        self.cap = lib.create_capture()
        self.name_buffer = create_string_buffer(255);
        self.buffer = None
        self.have_devices = False
        self.size = None
        self.real_size = None
        self.info = None

    def __del__(self):
        if not self.buffer is None:
            del self.buffer
        del self.name_buffer
        self.destroy_capture()

    def get_devices(self):
        self.have_devices = True
        return self.lib.get_devices(self.cap)

    def get_device(self, device_number):
        self.lib.get_device(self.cap, device_number, self.name_buffer, 255)
        name_str = str(self.name_buffer.value.decode('utf8'))
        return name_str

    def get_info(self):
        json_length = self.lib.get_json_length(self.cap);
        json_buffer = create_string_buffer(json_length)
        self.lib.get_json(self.cap, json_buffer, json_length);
        json_text = str(json_buffer.value.decode('utf8'))
        self.info = json.loads(json_text)
        return self.info

    def capture_device(self, cam, width, height, fps):
        if not self.have_devices:
            self.get_devices()
        ret = self.lib.capture_device(self.cap, cam, width, height, fps) == 1
        if ret:
            self.width = self.get_width()
            self.height = self.get_height()
            self.flipped = self.get_flipped()
            self.colorspace = self.get_colorspace()
            self.size = self.width * self.height * 4
            self.buffer = create_frame_buffer(self.width, self.height)
        else:
            self.size = None
            self.real_size = None
        return ret;

    def capture_device_by_dcap(self, cam, dcap, width, height, fps):
        if not self.have_devices:
            self.get_devices()
        fps = int(10000000 / fps)
        ret = self.lib.capture_device_by_dcap(self.cap, cam, dcap, width, height, fps) == 1
        if ret:
            self.width = self.get_width()
            self.height = self.get_height()
            self.flipped = self.get_flipped()
            self.colorspace = self.get_colorspace()
            self.size = self.width * self.height * 4 * 4
            self.buffer = create_frame_buffer(self.width, self.height)
        else:
            self.size = None
            self.real_size = None
        return ret;

    def capture_device_default(self, cam):
        if not self.have_devices:
            self.get_devices()
        ret = self.lib.capture_device_default(self.cap, cam) == 1
        if ret:
            self.width = self.get_width()
            self.height = self.get_height()
            self.flipped = self.get_flipped()
            self.colorspace = self.get_colorspace()
            self.size = self.width * self.height * 4 * 4
            self.buffer = create_frame_buffer(self.width, self.height)
        else:
            self.size = None
            self.real_size = None
        return ret;

    def get_width(self):
        return self.lib.get_width(self.cap)

    def get_height(self):
        return self.lib.get_height(self.cap)

    def get_fps(self):
        return self.lib.get_fps(self.cap)

    def get_flipped(self):
        return self.lib.get_flipped(self.cap) != 0

    def get_colorspace(self):
        colorspace = self.lib.get_colorspace(self.cap)
        if colorspace == 0:
            colorspace = self.lib.get_colorspace_internal(self.cap)
        return colorspace

    def get_colorspace_internal(self):
        return self.lib.get_colorspace_internal(self.cap)

    def capturing(self):
        return self.lib.capturing(self.cap) == 1

    def get_frame(self, timeout):
        if self.size is None:
            return None
        self.real_size = self.lib.get_frame(self.cap, timeout, self.buffer, self.size)
        img = np.frombuffer(self.buffer, dtype=np.uint8)[0:self.real_size]
        if self.colorspace in [100, 101]:
            if self.real_size == self.height * self.width * 4:
                img = cv2.cvtColor(img.reshape((self.height,self.width,4)), cv2.COLOR_BGRA2BGR)
            elif self.real_size == self.height * self.width * 3:
                img = img.reshape((self.height,self.width,3))
            else:
                return None
            if not self.flipped:
                img = cv2.flip(img, 0)
            return img
        elif self.colorspace == 200:
            img = cv2.cvtColor(img.reshape((3 * self.height // 2,self.width,1)), cv2.COLOR_YUV2BGR_I420)
        elif self.colorspace == 201:
            img = cv2.cvtColor(img.reshape((3 * self.height // 2,self.width,1)), cv2.COLOR_YUV2BGR_NV12)
        elif self.colorspace == 202:
            img = cv2.cvtColor(img.reshape((3 * self.height // 2,self.width,1)), cv2.COLOR_YUV2BGR_YV12)
        elif self.colorspace == 203:
            img = cv2.cvtColor(img.reshape((self.height,self.width,1)), cv2.COLOR_GRAY2BGR)
        elif self.colorspace == 300:
            img = cv2.cvtColor(img.reshape((self.height,self.width,2)), cv2.COLOR_YUV2BGR_YVYU)
        elif self.colorspace == 301:
            img = cv2.cvtColor(img.reshape((self.height,self.width,2)), cv2.COLOR_YUV2BGR_YUY2)
        elif self.colorspace == 302:
            img = cv2.cvtColor(img.reshape((self.height,self.width,2)), cv2.COLOR_YUV2BGR_UYVY)
        elif self.colorspace == 303:
            return None
        elif self.colorspace == 400:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR);
        if self.flipped:
            img = cv2.flip(img, 0)
        return img

    def stop_capture(self):
        self.size = None
        self.real_size = None
        return self.lib.stop_capture(self.cap)

    def destroy_capture(self):
        if self.cap is None:
            return 0
        ret = self.lib.destroy_capture(self.cap)
        self.cap = None
        self.size = None
        self.real_size = None
        return ret

if __name__ == "__main__":
    cam = 0
    width = 1280
    height = 720
    fps = 30
    if len(sys.argv) > 1:
        cam = int(sys.argv[1])
    if len(sys.argv) > 2:
        width = int(sys.argv[2])
    if len(sys.argv) > 3:
        height = int(sys.argv[3])
    if len(sys.argv) > 4:
        fps = int(sys.argv[4])

    cap = DShowCapture()
    devices = cap.get_devices()
    print("Devices: ", devices)
    for i in range(devices):
        print(f"{i} {cap.get_device(i)}")

    print(f"Capturing: {cap.capture_device(cam, width, height, fps)}")
    width = cap.get_width()
    height = cap.get_height()
    flipped = cap.get_flipped()
    print(f"Width: {width} Height: {height} FPS: {cap.get_fps()} Flipped: {flipped} Colorspace: {cap.get_colorspace()} Internal: {cap.get_colorspace_internal()}")
    
    while True:
        img = cap.get_frame(1000)
        if not img is None:
            cv2.imshow("DShowCapture", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)
        else:
            print("Lost frame")

    cap.stop_capture()
    cap.destroy_capture()
    del cap