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
bm_lib = None
bm_options = None
bm_enabled = False

def set_bm_enabled(v):
    global bm_enabled
    bm_enabled = v

def set_options(str):
    global bm_options
    bm_options = str

def create_frame_buffer(width, height, factor):
    buffer = bytearray(width * height * 4 * factor)
    char_array = c_char * len(buffer)
    return char_array.from_buffer(buffer)

class DShowCapture():
    def __init__(self):
        global lib
        global bm_lib
        global bm_options
        global bm_enabled
        if lib is None or bm_lib is None:
            if platform.architecture()[0] == '32bit':
                dll_path = resolve(os.path.join("dshowcapture", "dshowcapture_x86.dll"))
                lib = cdll.LoadLibrary(dll_path)
                if bm_enabled:
                    dll_path = resolve(os.path.join("dshowcapture", "libminibmcapture32.dll"))
                    bm_lib = cdll.LoadLibrary(dll_path)
            else:
                dll_path = resolve(os.path.join("dshowcapture", "dshowcapture_x64.dll"))
                lib = cdll.LoadLibrary(dll_path)
                if bm_enabled:
                    dll_path = resolve(os.path.join("dshowcapture", "libminibmcapture64.dll"))
                    bm_lib = cdll.LoadLibrary(dll_path)

            # DirectShow
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

            # Blackmagic
            if bm_enabled:
                bm_lib.start_capture_single.argtypes = [c_int, c_int, c_void_p]
                bm_lib.read_frame_bgra32_blocking.argtypes = [c_char_p, c_int]
                bm_lib.stop_capture_single.argtypes = []
                bm_lib.get_json_length.argtypes = []
                bm_lib.get_json.argtypes = [c_char_p, c_int]
                if bm_options is not None:
                    bm_lib.set_options.argtypes = [c_char_p]
                    bm_lib.set_options(bm_options.encode())
        self.lib = lib
        if bm_enabled:
            self.bm_lib = bm_lib
        self.cap = lib.create_capture()
        self.name_buffer = create_string_buffer(255);
        self.buffer = None
        self.have_devices = False
        self.size = None
        self.real_size = None
        self.info = None
        self.type = None

    def __del__(self):
        if self.buffer is not None:
            del self.buffer
        del self.name_buffer
        self.destroy_capture()

    def get_devices(self):
        self.have_devices = True
        return self.lib.get_devices(self.cap)

    def get_device(self, device_number):
        self.lib.get_device(self.cap, device_number, self.name_buffer, 255)
        name_str = str(self.name_buffer.value.decode('utf8', 'surrogateescape'))
        return name_str

    def get_info(self):
        global bm_enabled
        # DirectShow
        json_length = self.lib.get_json_length(self.cap);
        json_buffer = create_string_buffer(json_length)
        self.lib.get_json(self.cap, json_buffer, json_length);
        json_text = str(json_buffer.value.decode('utf8', 'surrogateescape'))
        self.info = json.loads(json_text)
        for cam in self.info:
            cam["type"] = "DirectShow"
            cam["index"] = cam["id"]
        
        # Blackmagic
        if bm_enabled:
            json_length = self.bm_lib.get_json_length();
            json_buffer = create_string_buffer(json_length)
            self.bm_lib.get_json(json_buffer, json_length);
            json_text = str(json_buffer.value.decode('utf8', 'surrogateescape'))
            bm_info = json.loads(json_text)
            dshow_len = len(self.info)
            for cam in bm_info:
                cam["type"] = "Blackmagic"
                cam["index"] = cam["id"] + dshow_len
            self.info.extend(bm_info)

        return self.info

    def capture_device(self, cam, width, height, fps):
        if not self.have_devices:
            self.get_devices()
        ret = self.lib.capture_device(self.cap, cam, width, height, fps) == 1
        if ret:
            self.type = "DirectShow"
            self.width = self.get_width()
            self.height = self.get_height()
            self.fps = self.get_fps()
            self.flipped = self.get_flipped()
            self.colorspace = self.get_colorspace()
            self.colorspace_internal = self.get_colorspace_internal()
            self.size = self.width * self.height * 4
            self.buffer = create_frame_buffer(self.width, self.height, 4)
        else:
            self.size = None
            self.real_size = None
        return ret;

    def capture_device_by_dcap(self, cam, dcap, width, height, fps):
        global bm_enabled
        if not self.have_devices:
            self.get_devices()
        if self.info is None:
            self.get_info()
        fps = int(10000000 / fps)
        ret = False
        if self.info[cam]['type'] == "DirectShow":
            ret = self.lib.capture_device_by_dcap(self.cap, cam, dcap, width, height, fps) == 1
        elif bm_enabled and self.info[cam]['type'] == "Blackmagic":
            ret = self.bm_lib.start_capture_single(self.info[cam]['id'], self.info[cam]['caps'][dcap]['bmModecode'], None) == 1
        if ret:
            self.type = self.info[cam]['type']
            if self.type == "DirectShow":
                self.width = self.get_width()
                self.height = self.get_height()
                self.fps = self.get_fps()
                self.flipped = self.get_flipped()
                self.colorspace = self.get_colorspace()
                self.colorspace_internal = self.get_colorspace_internal()
                self.size = self.width * self.height * 4 * 4
                self.buffer = create_frame_buffer(self.width, self.height, 4)
            elif bm_enabled and self.type == "Blackmagic":
                self.width = self.info[cam]['caps'][dcap]['minCX']
                self.height = self.info[cam]['caps'][dcap]['minCY']
                self.fps = int(10000000 / self.info[cam]['caps'][dcap]['minInterval'])
                self.flipped = True
                self.colorspace = 101
                self.colorspace_internal = self.colorspace
                self.size = self.width * self.height * 4
                self.buffer = create_frame_buffer(self.width, self.height, 1)
                self.real_size = self.size
        else:
            self.size = None
            self.real_size = None
        return ret;

    def capture_device_default(self, cam):
        if not self.have_devices:
            self.get_devices()
        ret = self.lib.capture_device_default(self.cap, cam) == 1
        if ret:
            self.type = "DirectShow"
            self.width = self.get_width()
            self.height = self.get_height()
            self.fps = self.get_fps()
            self.flipped = self.get_flipped()
            self.colorspace = self.get_colorspace()
            self.colorspace_internal = self.get_colorspace_internal()
            self.size = self.width * self.height * 4 * 4
            self.buffer = create_frame_buffer(self.width, self.height, 4)
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
        if self.type == "Blackmagic":
            return 1
        return self.lib.capturing(self.cap) == 1

    def get_frame(self, timeout):
        global bm_enabled
        if self.size is None:
            return None
        if self.type == "DirectShow":
            self.real_size = self.lib.get_frame(self.cap, timeout, self.buffer, self.size)
        elif bm_enabled and self.type == "Blackmagic":
            self.bm_lib.read_frame_bgra32_blocking(self.buffer, self.size)
        else:
            return None
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
        elif self.colorspace == 200 and self.real_size == (3 * self.height // 2) * self.width:
            img = cv2.cvtColor(img.reshape((3 * self.height // 2,self.width,1)), cv2.COLOR_YUV2BGR_I420)
        elif self.colorspace == 201 and self.real_size == (3 * self.height // 2) * self.width:
            img = cv2.cvtColor(img.reshape((3 * self.height // 2,self.width,1)), cv2.COLOR_YUV2BGR_NV12)
        elif self.colorspace == 202 and self.real_size == (3 * self.height // 2) * self.width:
            img = cv2.cvtColor(img.reshape((3 * self.height // 2,self.width,1)), cv2.COLOR_YUV2BGR_YV12)
        elif self.colorspace == 203 and self.real_size == self.height * self.width:
            img = cv2.cvtColor(img.reshape((self.height,self.width,1)), cv2.COLOR_GRAY2BGR)
        elif self.colorspace == 300 and self.real_size == self.height * self.width * 2:
            img = cv2.cvtColor(img.reshape((self.height,self.width,2)), cv2.COLOR_YUV2BGR_YVYU)
        elif self.colorspace == 301 and self.real_size == self.height * self.width * 2:
            img = cv2.cvtColor(img.reshape((self.height,self.width,2)), cv2.COLOR_YUV2BGR_YUY2)
        elif self.colorspace == 302 and self.real_size == self.height * self.width * 2:
            img = cv2.cvtColor(img.reshape((self.height,self.width,2)), cv2.COLOR_YUV2BGR_UYVY)
        elif self.colorspace == 303:
            return None
        elif self.colorspace == 400 and self.real_size > 4:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR);
        else:
            return None
        if self.flipped:
            img = cv2.flip(img, 0)
        return img

    def stop_capture(self):
        global bm_enabled
        self.size = None
        self.real_size = None
        if self.type == "DirectShow":
            return self.lib.stop_capture(self.cap)
        elif bm_enabled and self.type == "Blackmagic":
            return self.bm_lib.stop_capture_single() != 0
        return True

    def destroy_capture(self):
        if self.cap is None:
            return 0
        self.stop_capture()
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
        if img is not None:
            cv2.imshow("DShowCapture", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)
        else:
            print("Lost frame")

    cap.stop_capture()
    cap.destroy_capture()
    del cap