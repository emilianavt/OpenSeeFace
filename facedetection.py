import os
import numpy as np
import cv2
import onnxruntime


#kinda split on how to handle this one
#It's the cause of all of my slowest frames
#but it's not run often
#and I don't know how to make it faster
class FaceDetector():
    def __init__(self, detection_threshold=0.6):

        model_base_path = os.path.join(os.path.dirname(__file__), os.path.join("models"))
        providersList = onnxruntime.capi._pybind_state.get_available_providers()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.detection = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_detection_opt.onnx"), sess_options=options, providers=providersList)

        self.mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
        self.std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
        self.detection_threshold = detection_threshold

    def detect_faces(self, frame):

            im = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR) * self.std + self.mean
            im = np.expand_dims(im, 0)
            im = np.transpose(im, (0,3,1,2))

            outputs, maxpool = self.detection.run([], {'input': im})

            outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
            detections = np.flip(np.argsort(outputs[0,0].flatten()))
            results = []
            if len(detections) == 0:
                return results

            x = detections[0] % 56
            y = detections[0] // 56
            if outputs[0, 0, y, x] < self.detection_threshold:
                return results

            r = outputs[0, 1, y, x] * 112.
            results.append(((x * 4) - r, (y * 4) - r, r*2,r*2))
            results = np.array(results).astype(np.float32)
            results[:, [0,2]] *= frame.shape[1] / 224.
            results[:, [1,3]] *= frame.shape[0] / 224.
            return results
