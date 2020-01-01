import os
import numpy as np
import cv2
import onnxruntime
import time
import queue
import threading

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f

def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

def intersects(r1, r2):
    r1_x1, r1_y1, r1_x2, r1_y2 = r1
    r1_x2 += r1_x1
    r1_y2 += r1_y1
    r2_x1, r2_y1, r2_x2, r2_y2 = r2
    r2_x2 += r2_x1
    r2_y2 += r2_y1
    return not (r1_x1 > r2_x2 or r1_x2 < r2_x1 or r1_y1 > r2_y2 or r1_y2 < r2_y1)

def group_rects(rects):
    rect_groups = {}
    for rect in rects:
        rect_groups[str(rect)] = [-1, -1, []]
    group_id = 0
    for i, rect in enumerate(rects):
        name = str(rect)
        group = group_id
        group_id += 1
        if rect_groups[name][0] < 0:
            rect_groups[name] = [group, -1, []]
        else:
            group = rect_groups[name][0]
        for j, other_rect in enumerate(rects): 
            if i == j:
                continue;
            inter = intersects(rect, other_rect)
            if intersects(rect, other_rect):
                rect_groups[str(other_rect)] = [group, -1, []]
    return rect_groups

def logit(p):
    if p >= 1.0:
        p = 0.9999999
    if p <= 0.0:
        p = 0.0000001
    p = p/(1-p)
    return float(np.log(p) / 16.0)

def matrix_to_quaternion(m):
    t = 0.0
    q = [0.0, 0.0, 0, 0.0]
    if m[2,2] < 0:
        if m[0,0] > m[1,1]:
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = [t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]]
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = [m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]]
    else:
        if m[0,0] < -m[1,1]:
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = [m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]]
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = [m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t]
    q = np.array(q, np.float32) * 0.5 / np.sqrt(t)
    return q

def worker_thread(session, input, crop_info, queue, input_name, idx, tracker):
    output = session.run([], {input_name: input})[0]
    conf, lms = tracker.landmarks(output[0], crop_info)
    queue.put((session, conf, lms, crop_info, idx))

class Tracker():
    def __init__(self, width, height, model_type=1, threshold=0.4, max_faces=1, discard_after=5, scan_every=3, bbox_growth=0.0, max_threads=4, silent=False, pnp_quality=1, model_dir=None):
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max_threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.model_type = model_type
        self.models = ["snv2_opt.onnx", "snv2_opt_b.onnx", "snv2_opt_fast.onnx"]
        model = self.models[self.model_type]
        model_base_path = resolve(os.path.join("models"))
        if model_dir is None:
            if not os.path.exists(model_base_path):
                model_base_path = resolve(os.path.join("..", "models"))
        else:
            model_base_path = model_dir
        
        # Single face instance with multiple threads
        self.session = onnxruntime.InferenceSession(os.path.join(model_base_path, model), sess_options=options)
        
        # Multiple faces with single threads
        self.sessions = []
        self.max_workers = min(max_threads, max_faces)
        extra_threads = max_threads % self.max_workers
        for i in range(self.max_workers):
            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = max_threads // self.max_workers
            if options.intra_op_num_threads < 1:
                options.intra_op_num_threads = 1
            elif i < extra_threads:
                options.intra_op_num_threads += 1
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions.append(onnxruntime.InferenceSession(os.path.join(model_base_path, model), sess_options=options))
        self.input_name = self.session.get_inputs()[0].name

        self.faceCascade = cv2.CascadeClassifier()
        self.faceCascade.load(os.path.join(model_base_path, "haarcascade_frontalface_alt2.xml"))
        self.faces = []

        # Image normalization constants
        self.mean = np.float32(np.array([0.485, 0.456, 0.406]))
        self.std = np.float32(np.array([0.229, 0.224, 0.225]))
        self.mean = self.mean / self.std
        self.std = self.std * 255.0

        # PnP solving
        self.high_quality_3d = False if pnp_quality == 0 else True
        self.rotation = None
        self.translation = None
        self.face_3d = np.array([
            [0.50194, 0.28030, -0.49561],
            [0.48377, 0.15031, -0.50000],
            [0.46233, 0.01987, -0.49781],
            [0.43017, -0.11278, -0.49123],
            [0.38130, -0.25112, -0.46491],
            [0.31761, -0.35653, -0.40132],
            [0.24840, -0.44335, -0.30482],
            [0.15900, -0.51157, -0.22588],
            [0.00194, -0.55490, -0.16009],
            [-0.15900, -0.51157, -0.22588],
            [-0.24840, -0.44335, -0.30482],
            [-0.31761, -0.35653, -0.40132],
            [-0.38130, -0.25112, -0.46491],
            [-0.43017, -0.11278, -0.49123],
            [-0.46233, 0.01987, -0.49781],
            [-0.48377, 0.15031, -0.50000],
            [-0.50194, 0.28030, -0.49561],
            [0.44537, 0.38375, -0.22807],
            [0.38759, 0.42826, -0.18421],
            [0.32259, 0.44510, -0.12500],
            [0.24917, 0.44510, -0.07895],
            [0.19500, 0.42104, -0.06798],
            [-0.19500, 0.42104, -0.06798],
            [-0.24917, 0.44510, -0.07895],
            [-0.32259, 0.44510, -0.12500],
            [-0.38759, 0.42826, -0.18421],
            [-0.44537, 0.38375, -0.22807],
            [0.00000, 0.25503, -0.08553],
            [0.00000, 0.16240, -0.04825],
            [0.00000, 0.07218, -0.01535],
            [0.00000, 0.00000, 0.0000],
            [0.08353, -0.02165, -0.12281],
            [0.04738, -0.05019, -0.08991],
            [0.00000, -0.05895, -0.07456],
            [-0.04738, -0.05019, -0.08991],
            [-0.08353, -0.02165, -0.12281],
            [0.32981, 0.28270, -0.19956],
            [0.28528, 0.30435, -0.17544],
            [0.23111, 0.29714, -0.16009],
            [0.17213, 0.27789, -0.14474],
            [0.22870, 0.24781, -0.16009],
            [0.30093, 0.24661, -0.17544],
            [-0.17213, 0.27789, -0.14474],
            [-0.23111, 0.29714, -0.16009],
            [-0.28528, 0.30435, -0.17544],
            [-0.32981, 0.28270, -0.19956],
            [-0.30093, 0.24661, -0.17544],
            [-0.22870, 0.24781, -0.16009],
            [0.10871, -0.14282, -0.16667],
            [0.04809, -0.12118, -0.12500],
            [0.00000, -0.14931, -0.10307],
            [-0.04809, -0.12118, -0.12500],
            [-0.10871, -0.14282, -0.16667],
            [-0.11348, -0.24018, -0.16228],
            [-0.05719, -0.28129, -0.14474],
            [0.00000, -0.29644, -0.10965],
            [0.05719, -0.28129, -0.14474],
            [0.11348, -0.24018, -0.16228],
            [0.14984, -0.16878, -0.19956],
            [0.06736, -0.17922, -0.14254],
            [0.00000, -0.18861, -0.10965],
            [-0.06736, -0.17922, -0.14254],
            [-0.14984, -0.16878, -0.19956],
            [-0.04980, -0.21723, -0.14035],
            [0.00000, -0.21201, -0.11404],
            [0.04980, -0.21723, -0.14035],
        ], np.float32) * np.array([1.0, 1.0, 1.15])
        
        self.contour = np.empty((21 if self.high_quality_3d else 18, 3))
        self.contour[0:17] = self.face_3d[0:17]
        if self.high_quality_3d:
            self.contour[17:21] = self.face_3d[27:31]
        else:
            self.contour[17] = self.face_3d[30]
        self.camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
        self.inverse_camera = np.linalg.inv(self.camera)
        self.dist_coeffs = np.zeros((4,1))

        self.width = width
        self.height = height
        self.threshold = threshold
        self.max_faces = max_faces
        self.max_threads = max_threads
        self.discard = 0
        self.discard_after = discard_after
        self.detected = 0
        self.wait_count = 0
        self.scan_every = scan_every
        self.bbox_growth = bbox_growth
        self.silent = silent
        self.res = 224. if self.model_type < 2 else 112.
        self.res_i = int(self.res)

    def landmarks(self, tensor, crop_info):
        crop_x1, crop_y1, scale_x, scale_y = crop_info
        avg_conf = 0
        lms = []
        for i in range(0, 66):
            m = int(tensor[i].argmax())
            x = m // 28
            y = m % 28
            conf = float(tensor[i][x,y])
            avg_conf = avg_conf + conf
            off_x = 0
            off_y = 0
            if self.model_type > 0:
                off_x = self.res * ((1. * logit(tensor[66 + i][y, x])) - 0.0)
                off_y = self.res * ((1. * logit(tensor[66 * 2 + i][y, x])) - 0.0)
            else:
                off_x = self.res * ((2. * float(tensor[66 + i][y, x])) - 1.0)
                off_y = self.res * ((2. * float(tensor[66 * 2 + i][y, x])) - 1.0)
            x = crop_y1 + scale_y * (self.res * (float(x) / 28.) + off_x)
            y = crop_x1 + scale_x * (self.res * (float(y) / 28.) + off_y)
            lms.append((x,y,conf))
        avg_conf = avg_conf / 66.
        return (avg_conf, lms)

    def estimateDepth(self, lms):
        image_pts = np.empty((21 if self.high_quality_3d else 18, 2), np.float32)
        image_pts[0:17] = np.array(lms)[0:17,0:2]
        if (self.high_quality_3d):
            image_pts[17:21] = np.array(lms)[27:31,0:2]
        else:
            image_pts[17] = np.array(lms)[30,0:2]

        success = False
        if self.max_faces < 2 and not self.rotation is None:
            success, self.rotation, self.translation = cv2.solvePnP(self.contour, image_pts, self.camera, self.dist_coeffs, useExtrinsicGuess=True, rvec=np.transpose(self.rotation), tvec=np.transpose(self.translation), flags=cv2.SOLVEPNP_EPNP)
        else:
            success, self.rotation, self.translation = cv2.solvePnP(self.contour, image_pts, self.camera, self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

        rotation = self.rotation
        translation = self.translation

        if not success:
            self.rotation = np.array([0.0, 0.0, 0.0], np.float32)
            self.translation = np.array([0.0, 0.0, 0.0], np.float32)
            return False, np.zeros(4), 0.0, []
        else:
            self.rotation = np.transpose(self.rotation)
            self.translation = np.transpose(self.translation)

        rmat, _ = cv2.Rodrigues(rotation)
        inverse_rotation = np.linalg.inv(rmat)
        pts_3d = []
        pnp_error = 0.0
        for i, pt in enumerate(self.face_3d):
            reference = rmat.dot(pt)
            reference = reference + self.translation
            reference = self.camera.dot(reference[0])
            depth = reference[2]
            if i < 17 or i == 30:
                reference = reference / depth
                e1 = lms[i][0] - reference[0]
                e2 = lms[i][1] - reference[1]
                pnp_error += e1*e1 + e2*e2
            pt_2d = np.array([lms[i][0] * depth, lms[i][1] * depth, depth], np.float32)
            pt_2d = self.inverse_camera.dot(pt_2d) 
            pt_2d = pt_2d - self.translation
            pt_2d = inverse_rotation.dot(pt_2d[0])
            pts_3d.append(pt_2d)

        euler = cv2.RQDecomp3x3(rmat)[0]
        return True, matrix_to_quaternion(rmat), euler, np.sqrt(pnp_error / (2 * 17.0)), pts_3d

    def preprocess(self, im, crop):
        x1, y1, x2, y2 = crop
        im = np.float32(im[y1:y2, x1:x2,::-1]) # Crop and BGR to RGB
        im = cv2.resize(im, (self.res_i, self.res_i), interpolation=cv2.INTER_LINEAR) / self.std - self.mean
        #im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0
        #im = (im - mean) / std
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        if self.model_type > 1:
            im = im.mean(1)
            im = np.expand_dims(im, 1)
        return im

    def predict(self, frame, additional_faces=[]):
        start = time.time()
        im = frame

        duration_fd = 0.0
        duration_pp = 0.0
        duration_model = 0.0
        duration_pnp = 0.0

        new_faces = []
        new_faces.extend(self.faces)
        new_faces.extend(additional_faces)
        self.wait_count += 1
        if self.detected < self.max_faces:
            if self.detected == 0 or self.wait_count >= self.scan_every:
                start_fd = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                new_faces.extend(list(self.faceCascade.detectMultiScale(gray, 1.3, 4, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50))))
                duration_fd = 1000 * (time.time() - start_fd)
                self.wait_count = 0
        else:
            self.wait_count = 0

        if len(new_faces) < 1:
            duration = (time.time() - start) * 1000
            if not self.silent:
                print(f"Took {duration:.2f}ms")
            return []

        crops = []
        crop_info = []
        num_crops = 0
        for j, (x,y,w,h) in enumerate(new_faces):
            if self.model_type < 2:
                crop_x1 = x - int(w * 0.2)
                crop_y1 = y - int(h * 0.25)
                crop_x2 = x + w + int(w * 0.2)
                crop_y2 = y + h + int(h * 0.25)
            else:
                crop_x1 = x - int(w * 0.1)
                crop_y1 = y - int(h * 0.1)
                crop_x2 = x + w + int(w * 0.1)
                crop_y2 = y + h + int(h * 0.1)
            crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), self.width, self.height)
            crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), self.width, self.height)

            scale_x = float(crop_x2 - crop_x1) / self.res
            scale_y = float(crop_y2 - crop_y1) / self.res
            
            if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                continue

            start_pp = time.time()
            crop = self.preprocess(im, (crop_x1, crop_y1, crop_x2, crop_y2))
            duration_pp += 1000 * (time.time() - start_pp)
            crops.append(crop)
            crop_info.append((crop_x1, crop_y1, scale_x, scale_y))
            num_crops += 1

        groups = group_rects(new_faces)

        start_model = time.time()
        outputs = {}
        if num_crops == 1:
            output = self.session.run([], {self.input_name: crops[0]})[0]
            conf, lms = self.landmarks(output[0], crop_info[0])
            outputs[crop_info[0]] = (conf, lms, 0)
        else:
            started = 0
            results = queue.Queue()
            for i in range(min(num_crops, self.max_workers)):
                thread = threading.Thread(target=worker_thread, args=(self.sessions[started], crops[started], crop_info[started], results, self.input_name, started, self))
                started += 1
                thread.start()
            returned = 0
            while returned < num_crops:
                session, conf, lms, sample_crop_info, idx = results.get(True)
                outputs[sample_crop_info] = (conf, lms, idx)
                returned += 1
                if started < num_crops:
                    thread = threading.Thread(target=worker_thread, args=(session, crops[started], crop_info[started], results, self.input_name, started, self))
                    started += 1
                    thread.start()

        best_results = {}
        for crop in crop_info:
            conf, lms, i = outputs[crop]
            if conf < self.threshold:
                continue;
            group_id = groups[str(new_faces[i])][0]
            if not group_id in best_results:
                best_results[group_id] = [-1, []]
            if conf > self.threshold and best_results[group_id][0] < conf:
                best_results[group_id][0] = conf
                best_results[group_id][1] = lms
        
        sorted_results = sorted(best_results.values(), key=lambda x: x[0], reverse=True)
        duration_model = 1000 * (time.time() - start_model)

        results = []
        detected = []
        detections = 0
        for conf, lms in sorted_results:
            if detections >= self.max_faces:
                break;
            if conf > self.threshold:
                detections += 1
                start_pnp = time.time()
                success, quaternion, euler, pnp_error, pts_3d = self.estimateDepth(lms)
                duration_pnp += 1000 * (time.time() - start_pnp)
                results.append((conf, lms, success, pnp_error, quaternion, euler, self.rotation, self.translation, pts_3d))
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for (x,y,c) in lms:
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                w = max_x - min_x
                y = max_y - min_y
                detected.append((min_y, min_x, max_y - min_y, max_x - min_x))

        if len(detected) > 0:
            self.detected = len(detected)
            self.faces = detected
            self.discard = 0
        else:
            self.rotation = None
            self.translation = None
            self.detected = 0
            self.discard += 1
            if self.discard > self.discard_after:
                self.faces = []
            else:
                if self.bbox_growth > 0:
                    faces = []
                    for (x,y,w,h) in self.faces: 
                        x -= w * self.bbox_growth
                        y -= h * self.bbox_growth
                        w += 2 * w * self.bbox_growth
                        h += 2 * h * self.bbox_growth
                        faces.append((x,y,w,h))
                    self.faces = faces

        duration = (time.time() - start) * 1000
        if not self.silent:
            print(f"Took {duration:.2f}ms (detect: {duration_fd:.2f}ms, crop: {duration_pp:.2f}, track: {duration_model:.2f}ms, 3D points: {duration_pnp:.2f}ms)")

        return results
