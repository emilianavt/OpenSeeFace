import os
import numpy as np
import cv2
import onnxruntime
import time



class FaceDetector():
    def __init__(self, detection_threshold=0.6):

        self.mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
        self.std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
        self.detection_threshold = detection_threshold
        self.targetimageSize = [224, 224]
        self.image = None

    def detect_faces(self, frame, model):
            self.resizeImage(frame)

            outputs, _ = model.faceDetection.run([], {'input': self.image})
            outputs = outputs[0]
            faceLocation = np.argmax(outputs[0].flatten())
            x = faceLocation % 56
            y = faceLocation // 56

            if outputs[0, y, x] < self.detection_threshold:
                return None

            r = outputs[1, y, x] * 112.
            results= (((x * 4) - r, (y * 4) - r, r*2,r*2))
            results = np.array(results).astype(np.float32)
            results[[0,2]] *= frame.shape[1] / 224.
            results[[1,3]] *= frame.shape[0] / 224.

            return results

    def resizeImage(self, frame):
        image = cv2.resize(frame, self.targetimageSize, interpolation=cv2.INTER_LINEAR) * self.std + self.mean
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0,3,1,2))
        self.image = image



