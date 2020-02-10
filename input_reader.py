import re
import sys
import os
import cv2
import numpy as np
import escapi

def list_cameras():
    print("Available cameras:")

class VideoReader():
    def __init__(self, capture):
        self.cap = cv2.VideoCapture(capture)
        if self.cap is None:
            print("The video source cannot be opened")
            sys.exit(0)
    def is_open(self):
        return self.cap.isOpened()
    def is_ready(self):
        return True
    def read(self):
        return self.cap.read()
    def close(self):
        self.cap.release()

class CameraReader(VideoReader):
    def __init__(self, capture, width, height, fps):
        self.device = None
        self.width = width
        self.height = height
        self.fps = fps
        if os.name == 'nt':
            escapi.init()
            self.device = capture
            self.buffer = escapi.init_camera(self.device, self.width, self.height, self.fps)
            escapi.do_capture(self.device)
        else:
            super(CameraReader, self).__init__(capture)
            self.cap.set(3, width)
            self.cap.set(4, height)
    def is_open(self):
        if self.device is None:
            return super(CameraReader, self).is_open()
        return True
    def is_ready(self):
        if self.device is None:
            return super(CameraReader, self).is_ready()
        return escapi.is_capture_done(self.device)
    def read(self):
        if self.device is None:
            return super(CameraReader, self).read()
        if escapi.is_capture_done(self.device):
            image = escapi.read(self.device, self.width, self.height, self.buffer)
            escapi.do_capture(self.device)
            return True, image
        else:
            return False, None
    def close(self):
        if self.device is None:
            super(CameraReader, self).close()
        else:
            escapi.deinit_camera(self.device)

class RawReader:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        
        if self.width < 1 or self.height < 1:
            print("No acceptable size was given for reading raw RGB frames.")
            sys.exit(0)

        self.len = self.width * self.height * 3
        self.open = True
    def is_open(self):
        return self.open
    def is_ready(self):
        return True
    def read(self):
        frame = bytearray()
        read_bytes = 0
        while read_bytes < self.len:
            bytes = sys.stdin.buffer.read(self.len)
            read_bytes += len(bytes)
            frame.extend(bytes)
        return True, np.frombuffer(frame, dtype=np.uint8).reshape((self.height, self.width, 3))
    def close(self):
        self.open = False

class InputReader():
    def __init__(self, capture, raw_rgb, width, height, fps):
        self.reader = None
        try:
            if raw_rgb > 0:
                self.reader = RawReader(width, height)
            elif os.path.exists(capture):
                self.reader = VideoReader(capture)
            elif capture == str(int(capture)):
                self.reader = CameraReader(int(capture), width, height, fps)
        except Exception as e:
            print("Error: " + str(e))

        if self.reader is None or not self.reader.is_open():
            print("There was no valid input.")
            sys.exit(0)
    def is_open(self):
        return self.reader.is_open()
    def is_ready(self):
        return self.reader.is_ready()
    def read(self):
        return self.reader.read()
    def close(self):
        self.reader.close()
