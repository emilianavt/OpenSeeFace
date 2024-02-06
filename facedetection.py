import os
import numpy as np
import cv2
import onnxruntime
import time


#kinda split on how to handle this one
#It's the cause of all of my slowest frames
#but it's not run often
#and I don't know how to make it faster

class FaceDetectionModel():
    def __init__(self):
        model_base_path = os.path.join(os.path.dirname(__file__), os.path.join("models"))
        providersList = onnxruntime.capi._pybind_state.get_available_providers()

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.detection = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_detection_opt.onnx"), sess_options=options, providers=providersList)

    def run(self, image):
        outputs, maxpool = self.detection.run([], {'input': image})
        return outputs[0]




class FaceDetector():
    def __init__(self, detection_threshold=0.6):
        self.model = FaceDetectionModel()

        self.mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
        self.std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
        self.detection_threshold = detection_threshold
        self.targetimageSize = [224, 224]
        self.image = None

    def detect_faces(self, frame):
            self.resizeImage(frame)

            outputs = self.model.run(self.image)
            faceLocation = np.argmax(outputs[0].flatten())
            x = faceLocation % 56
            y = faceLocation // 56
            if outputs[0, y, x] < self.detection_threshold:
                return []

            r = outputs[1, y, x] * 112.
            results= (((x * 4) - r, (y * 4) - r, r*2,r*2))
            results = np.array(results).astype(np.float32)
            results[ [0,2]] *= frame.shape[1] / 224.
            results[ [1,3]] *= frame.shape[0] / 224.
            return results

    def resizeImage(self, frame):
        image = cv2.resize(frame, self.targetimageSize, interpolation=cv2.INTER_LINEAR) * self.std + self.mean
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0,3,1,2))
        self.image = image



