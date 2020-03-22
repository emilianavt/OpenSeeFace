import sys
import numpy as np
import cv2
import onnxruntime
import time
import queue
import threading
import json
import copy

def py_cpu_nms(dets, thresh):
    """ Pure Python NMS baseline.
        Copyright (c) 2015 Microsoft
        Licensed under The MIT License
        Written by Ross Girshick
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def decode(loc, priors, variances):
    data = (
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    )
    boxes = np.concatenate(data, 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def worker_thread(rfd, frame):
    results = rfd.detect_retina(frame, is_background=True)
    rfd.results.put(results, False)
    rfd.finished = True
    rfd.running = False

class RetinaFaceDetector():
    def __init__(self, model_path="models/retinaface_640x640_opt.onnx", json_path="models/priorbox_640x640.json", threads=4, min_conf=0.4, nms_threshold=0.4, top_k=1, res=(640, 640)):
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(model_path, sess_options=options)
        self.res_w, self.res_h = res
        with open(json_path, "r") as prior_file:
            self.priorbox = np.array(json.loads(prior_file.read()))
        self.min_conf = min_conf
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.finished = False
        self.running = False
        self.results = queue.Queue()

    def detect_retina(self, frame, is_background=False):
        h, w, _ = frame.shape
        im = None
        im = cv2.resize(frame, (self.res_w, self.res_h), interpolation=cv2.INTER_LINEAR)
        resize_w = w / self.res_w
        resize_w = 1 / resize_w
        resize_h = h / self.res_h
        resize_h = 1 / resize_h
        im = np.float32(im)
        scale = np.array((self.res_w / resize_w, self.res_h / resize_h, self.res_w / resize_w, self.res_h / resize_h))
        im -= (104, 117, 123)
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, 0)
        output = self.session.run([], {"input0": im})
        loc, conf = output[0][0], output[1][0]
        boxes = decode(loc, self.priorbox, [0.1, 0.2])
        boxes = boxes * scale
        scores = conf[:, 1]

        inds = np.where(scores > self.min_conf)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        dets = dets[:self.top_k, 0:4]
        dets[:, 2:4] = dets[:, 2:4] - dets[:, 0:2]

        if True:#is_background:
            upsize = dets[:, 2:4] * np.array([[0.15, 0.2]])
            dets[:, 0:2] -= upsize
            dets[:, 2:4] += upsize * 2

        return list(map(tuple, dets))

    def background_detect(self, frame):
        if self.running or self.finished:
            return
        self.running = True
        im = copy.copy(frame)
        thread = threading.Thread(target=worker_thread, args=(self, im))
        thread.start()

    def get_results(self):
        if self.finished:
            results = []
            try:
                while True:
                    detection = self.results.get(False)
                    results.append(detection)
            except:
                "No error"
            self.finished = False
            return list(*results)
        else:
            return []

if __name__== "__main__":
    retina = RetinaFaceDetector(top_k=40, min_conf=0.2)
    im = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    start = time.perf_counter()
    faces = retina.detect_retina(im)
    end = 1000 * (time.perf_counter() - start)
    print(f"Runtime: {end:.3f}ms")
    for (x,y,w,h) in faces:
        im = cv2.rectangle(im, (int(x),int(y)), (int(x+w),int(y+w)), (0,0,255), 1)
    cv2.imshow("Frame", im)
    while cv2.waitKey(0) & 0xff != ord('q'):
        ""
