import os

os.environ["OMP_NUM_THREADS"] = str(1)
import dshowcapture
import cv2
import multiprocessing
import time
import numpy as np
import math


#this is just here for conveience
def startProcess(frameQueue, faceQueue, fps, targetBrightness, width, height, mirrorInput):
    webcam = Webcam(frameQueue, faceQueue, fps, targetBrightness, width, height, mirrorInput)
    webcam.start()

class Webcam():
    def __init__(self, frameQueue, faceQueue, fps, targetBrightness, width, height, mirrorInput):
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)

        if height > 480:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.webcamBuffer = 2
        self.cap.set(38, self.webcamBuffer)
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.targetFrameTime = 1. / fps
        self.gamma = 0.7
        self.mirror = mirrorInput
        self.frameQueue = frameQueue #outgoing frames from the webcam
        self.faceQueue = faceQueue  #incoming face data for gamma calculation
        self.cameraLatency = 0
        self.brightnessFrame = None #a copy of the brightness channel for gamma calculations
        self.targetBrightness = targetBrightness    #the target average brightness of the face, used in gamma calculations
        self.width = width  #unused, but it seemed useful to have around
        self.height = height    #unused, but it seemed useful to have around
        self.ret = 0

    def start(self):
        while self.cap.isOpened():
            frameStart= time.perf_counter()
            totalFrameLatency = time.perf_counter()
            frame = self.getFrame()
            self.cameraLatency = time.perf_counter() - frameStart
            if self.ret:
                frame = self.applyGamma(frame)
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                #this line is where this process sits 90% of the time
                self.frameQueue.put([frame, self.cameraLatency, totalFrameLatency])
                if self.faceQueue.qsize() > 0:
                    self.updateGamma()
            sleepTime = self.targetFrameTime - (time.perf_counter() - frameStart)
            sleepTime = max(sleepTime, 0)
            time.sleep(sleepTime)

    def getFrame(self):
        #My webcam's gain tends to be noisy
        #and my gamma correction seems less noisy
        #so I tell the camera not to apply gain
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        #keeping track of how long it takes to get frames from the webcam
        #because that one line is the cause of 90% of late frames
        cameraStart = time.perf_counter()
        self.ret, frame = self.cap.read()
        self.cameraLatency = time.perf_counter() - cameraStart
        return(frame)

    #Applies a gamma curve to the frame
    #Uses a lookup table because that's way faster than actually calculating every pixel
    #Immensely improves tracking in low light situations
    def applyGamma(self, frame):
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        #saving a copy of the brightness channel so I can use it to adjust the gamma based on where the face is in that frame
        self.brightnessFrame = img_yuv[:,:,0].copy()
        #building a lookup table on the fly
        lookupTable = np.array(range(256))
        loopupTable = lookupTable/255
        lookupTable = np.power(loopupTable, self.gamma)*255
        img_yuv[:,:,0] = lookupTable[img_yuv[:,:,0]]
        #I convert the image to RBG here because it was getting repeatedly converted in the face tracking
        frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return frame


    #calculate the ideal gamma based on the brightness of the user's face
    #kind of winging it with the math, but it's a lot better than calculating based on the whole frame
    def updateGamma(self):
        face_center, face_radius = self.faceQueue.get()

        y1 = int(face_center[1] - face_radius[1])
        y2 = int(face_center[1] + face_radius[1])
        x1 = int(face_center[0] - face_radius[0])
        x2 = int(face_center[0] + face_radius[0])

        #I was doing random sampling of the image brightness
        #and that's probably brighter for the whole frame
        #but for just the face this is pretty fast
        averageBrightness = np.mean(self.brightnessFrame[y1:y2,x1:x2])/256
        self.gamma = math.log(self.targetBrightness, averageBrightness)
        return



