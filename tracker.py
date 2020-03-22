import os
import numpy as np
import math
import cv2
import onnxruntime
import time
import queue
import threading
import copy
from retinaface import RetinaFaceDetector
from remedian import remedian

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

def rotate(origin, point, a):
    a = -a
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy

def angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.arctan2(*(p2 - p1)[::-1])
    return (a % (2 * np.pi))

def compensate(p1, p2):
    a = angle(p1, p2)
    return rotate(p1, p2, a), a

def rotate_image(image, a, center):
    (h, w) = image.shape[:2]
    a = np.rad2deg(a)
    M = cv2.getRotationMatrix2D((center[0], center[1]), a, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

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

def logit(p, factor=16.0):
    if p >= 1.0:
        p = 0.9999999
    if p <= 0.0:
        p = 0.0000001
    p = p/(1-p)
    return float(np.log(p)) / float(factor)

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

class Feature():
    def __init__(self, threshold=0.15, alpha=0.2, hard_factor=0.15, decay=0.001):
        self.median = remedian()
        self.min = None
        self.max = None
        self.hard_min = None
        self.hard_max = None
        self.threshold = threshold
        self.alpha = alpha
        self.hard_factor = hard_factor
        self.decay = decay
        self.last = 0

    def update(self, x):
        new = self.update_state(x)
        filtered = self.last * self.alpha + new * (1 - self.alpha)
        self.last = filtered
        return filtered

    def update_state(self, x):
        self.median + x
        median = self.median.median()

        if self.min is None:
            if x < median and (median - x) / median > self.threshold:
                self.min = x
                self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
            return 0
        else:
            if x < self.min:
                self.min = x
                self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
        if self.max is None:
            if x > median and (x - median) / median > self.threshold:
                self.max = x
                self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1
            return 0
        else:
            if x > self.max:
                self.max = x
                self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1

        if self.min < self.hard_min:
            self.min = self.hard_min * self.decay + self.min * (1 - self.decay)
        if self.max > self.hard_max:
            self.max = self.hard_max * self.decay + self.max * (1 - self.decay)

        if x < median:
            return - (1 - (x - self.min) / (median - self.min))
        elif x > median:
            return (x - median) / (self.max - median)

        return 0

class FeatureExtractor():
    def __init__(self):
        self.eye_l = Feature()
        self.eye_r = Feature()
        self.eyebrow_updown_l = Feature()
        self.eyebrow_updown_r = Feature()
        self.eyebrow_quirk_l = Feature(threshold=0.05)
        self.eyebrow_quirk_r = Feature(threshold=0.05)
        self.eyebrow_steepness_l = Feature(threshold=0.05)
        self.eyebrow_steepness_r = Feature(threshold=0.05)
        self.mouth_corner_updown_l = Feature()
        self.mouth_corner_updown_r = Feature()
        self.mouth_corner_inout_l = Feature(threshold=0.02)
        self.mouth_corner_inout_r = Feature(threshold=0.02)
        self.mouth_open = Feature()
        self.mouth_wide = Feature(threshold=0.02)

    def align_points(self, a, b, pts):
        a = tuple(a)
        b = tuple(b)
        alpha = angle(a, b)
        alpha = np.rad2deg(alpha)
        if alpha >= 90:
            alpha = - (alpha - 180)
        if alpha <= -90:
            alpha = - (alpha + 180)
        alpha = np.deg2rad(alpha)
        aligned_pts = []
        for pt in pts:
            aligned_pts.append(np.array(rotate(a, pt, alpha)))
        return alpha, np.array(aligned_pts)

    def update(self, pts):
        features = {}

        norm_distance_x = np.mean([pts[0, 0] - pts[16, 0], pts[1, 0] - pts[15, 0]])
        norm_distance_y = np.mean([pts[27, 1] - pts[28, 1], pts[28, 1] - pts[29, 1], pts[29, 1] - pts[30, 1]])

        a1, f_pts = self.align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
        f = np.clip((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y, 0, None)
        features["eye_l"] = self.eye_l.update(f)

        a2, f_pts = self.align_points(pts[36], pts[39], pts[[37, 38, 41, 40]])
        f = np.clip((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y, 0, None)
        features["eye_r"] = self.eye_r.update(f)

        a3, _ = self.align_points(pts[0], pts[16], [])
        a4, _ = self.align_points(pts[31], pts[35], [])
        norm_angle = np.mean(list(map(np.rad2deg, [a1, a2, a3, a4])))

        a, f_pts = self.align_points(pts[22], pts[26], pts[[22, 23, 24, 25, 26]])
        features["eyebrow_steepness_l"] = self.eyebrow_steepness_l.update(-np.rad2deg(a) - norm_angle)
        f = (np.mean([pts[22, 1], pts[26, 1]]) - pts[27, 1]) / norm_distance_y
        features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f)
        f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
        features["eyebrow_quirk_l"] = self.eyebrow_quirk_l.update(f)

        a, f_pts = self.align_points(pts[17], pts[21], pts[[17, 18, 19, 20, 21]])
        features["eyebrow_steepness_r"] = self.eyebrow_steepness_r.update(np.rad2deg(a) - norm_angle)
        f = (np.mean([pts[17, 1], pts[21, 1]]) - pts[27, 1]) / norm_distance_y
        features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f)
        f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
        features["eyebrow_quirk_r"] = self.eyebrow_quirk_r.update(f)

        upper_mouth_line = np.mean([pts[49, 1], pts[50, 1], pts[51, 1]])
        center_line = np.mean([pts[50, 0], pts[60, 0], pts[27, 0], pts[30, 0], pts[64, 0], pts[55, 0]])

        f = (upper_mouth_line - pts[62, 1]) / norm_distance_y
        features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f)
        f = abs(center_line - pts[62, 0]) / norm_distance_x
        features["mouth_corner_inout_l"] = self.mouth_corner_inout_l.update(f)

        f = (upper_mouth_line - pts[58, 1]) / norm_distance_y
        features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f)
        f = abs(center_line - pts[58, 0]) / norm_distance_x
        features["mouth_corner_inout_r"] = self.mouth_corner_inout_r.update(f)

        f = (np.mean([pts[59, 1], pts[60, 1], pts[61, 1]]) - np.mean([pts[65, 1], pts[64, 1], pts[63, 1]])) / norm_distance_y
        features["mouth_open"] = self.mouth_open.update(f)

        f = abs(pts[58, 0] - pts[62, 0]) / norm_distance_x
        features["mouth_wide"] = self.mouth_wide.update(f)

        return features

class FaceInfo():
    def __init__(self, id, tracker):
        self.id = id
        self.frame_count = -1
        self.tracker = tracker
        self.contour_pts = [0,1,8,15,16,27,28,29,30,31,32,33,34,35,36,39,42,45]
        self.face_3d = copy.copy(self.tracker.face_3d)
        self.reset()
        self.alive = False
        self.coord = None

        self.limit_3d_adjustment = True
        self.update_count_delta = 75.
        self.update_count_max = 7500.

    def reset(self):
        self.alive = False
        self.conf = None
        self.lms = None
        self.eye_state = None
        self.rotation = None
        self.translation = None
        self.success = None
        self.quaternion = None
        self.euler = None
        self.pnp_error = None
        self.pts_3d = None
        self.eye_blink = None
        self.bbox = None
        self.pnp_error = 0
        self.features = FeatureExtractor()
        self.current_features = {}
        self.contour = np.zeros((21,3))
        self.update_counts = np.zeros((66,2))
        self.update_contour()

    def update(self, result, coord, frame_count):
        self.frame_count = frame_count
        if result is None:
            self.reset()
        else:
            self.conf, (self.lms, self.eye_state) = result
            self.coord = coord
            self.alive = True

    def update_contour(self):
        self.contour = np.array(self.face_3d[self.contour_pts])

    def normalize_pts3d(self, pts_3d):
        # Calculate angle using nose
        pts_3d[:, 0:2] -= pts_3d[30, 0:2]
        alpha = angle(pts_3d[30, 0:2], pts_3d[27, 0:2])
        alpha -= np.deg2rad(90)

        R = np.matrix([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        pts_3d[:, 0:2] = (pts_3d - pts_3d[30])[:, 0:2].dot(R) + pts_3d[30, 0:2]

        # Vertical scale
        pts_3d[:, 1] /= np.mean((pts_3d[27:30, 1] - pts_3d[28:31, 1]) / (self.tracker.face_3d[27:30, 1] - self.tracker.face_3d[28:31, 1]))

        # Horizontal scale
        pts_3d[:, 0] /= np.mean(np.abs(pts_3d[[0, 36, 42], 0] - pts_3d[[16, 39, 45], 0]) / np.abs(self.tracker.face_3d[[0, 36, 42], 0] - self.tracker.face_3d[[16, 39, 45], 0]))

        return pts_3d

    def adjust_3d(self):
        if self.conf < 0.4 or self.pnp_error > 300:
            return

        max_runs = 1
        eligible = np.arange(0, 66)
        changed_any = False
        update_type = -1
        d_o = np.ones((66,))
        d_c = np.ones((66,))
        for runs in range(max_runs):
            r = 1.0 + np.random.random_sample((66,3)) * 0.02 - 0.01
            if self.euler[0] > -165 and self.euler[0] < 145:
                continue
            elif self.euler[1] > -10 and self.euler[1] < 20:
                r[:, 2] = 1.0
                update_type = 0
            else:
                r[:, 0:2] = 1.0
                if self.euler[2] > 120 or self.euler[2] < 60:
                    continue
                # Enable only one side of the points, depending on direction
                elif self.euler[1] < -10:
                    update_type = 1
                    r[[0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 56, 57, 58, 59, 65], 2] = 1.0
                    eligible = [8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64]
                else:
                    update_type = 1
                    r[[9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 34, 35, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 61, 62, 63], 2] = 1.0
                    eligible = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 48, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65]

            if self.limit_3d_adjustment:
                eligible = np.nonzero(self.update_counts[:, update_type] < self.update_counts[:, abs(update_type - 1)] + self.update_count_delta)[0]
                if eligible.shape[0] <= 0:
                    break

            if runs == 0:
                updated = copy.copy(self.face_3d[0:66])
                o_projected = np.ones((66,2))
                o_projected[eligible] = np.squeeze(np.array(cv2.projectPoints(self.face_3d[eligible], self.rotation, self.translation, self.tracker.camera, self.tracker.dist_coeffs)[0]), 1)
            c = updated * r
            c_projected = np.zeros((66,2))
            c_projected[eligible] = np.squeeze(np.array(cv2.projectPoints(c[eligible], self.rotation, self.translation, self.tracker.camera, self.tracker.dist_coeffs)[0]), 1)
            changed = False

            d_o[eligible] = np.linalg.norm(o_projected[eligible] - self.lms[eligible, 0:2], axis=1)
            d_c[eligible] = np.linalg.norm(c_projected[eligible] - self.lms[eligible, 0:2], axis=1)
            indices = np.nonzero(d_c < d_o)[0]
            if indices.shape[0] > 0:
                if self.limit_3d_adjustment:
                    indices = np.intersect1d(indices, eligible)
                if indices.shape[0] > 0:
                    self.update_counts[indices, update_type] += 1
                    updated[indices] = c[indices]
                    o_projected[indices] = c_projected[indices]
                    changed = True
            changed_any = changed_any or changed

            if not changed:
                break

        if changed_any:
            # Update weighted by point confidence
            weights = np.zeros((66,3))
            weights[:, :] = self.lms[0:66, 2:3]
            weights[weights > 0.7] = 1.0
            weights = 1.0 - weights
            update_indices = np.arange(0, 66)
            if self.limit_3d_adjustment:
                update_indices = np.nonzero(self.update_counts[:, update_type] <= self.update_count_max)[0]
            self.face_3d[update_indices] = self.face_3d[update_indices] * weights[update_indices] + updated[update_indices] * (1. - weights[update_indices])
            self.update_contour()

        self.pts_3d = self.normalize_pts3d(self.pts_3d)
        self.current_features = self.features.update(self.pts_3d[:, 0:2])
        self.eye_blink = []
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_r"]), 1))
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_l"]), 1))


class Tracker():
    def __init__(self, width, height, model_type=3, threshold=0.4, max_faces=1, discard_after=5, scan_every=3, bbox_growth=0.0, max_threads=4, silent=False, model_dir=None, no_gaze=False, use_retinaface=False):
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max_threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.model_type = model_type
        self.models = [
            "snv2_opt_fast.onnx",
            "mnv3_opt_fast.onnx",
            "snv2_opt_b.onnx",
            "mnv3_opt_b.onnx"
        ]
        model = self.models[self.model_type]
        model_base_path = resolve(os.path.join("models"))
        if model_dir is None:
            if not os.path.exists(model_base_path):
                model_base_path = resolve(os.path.join("..", "models"))
        else:
            model_base_path = model_dir

        self.retinaface = RetinaFaceDetector(model_path=os.path.join(model_base_path, "retinaface_640x640_opt.onnx"), json_path=os.path.join(model_base_path, "priorbox_640x640.json"), threads=max_threads, top_k=max_faces, res=(640, 640))
        self.retinaface_scan = RetinaFaceDetector(model_path=os.path.join(model_base_path, "retinaface_640x640_opt.onnx"), json_path=os.path.join(model_base_path, "priorbox_640x640.json"), threads=2, top_k=max_faces, res=(640, 640))
        self.use_retinaface = use_retinaface

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

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max_threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.gaze_model = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_gaze32_split_opt.onnx"), sess_options=options)

        self.detection = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_detection_opt.onnx"), sess_options=options)
        self.faces = []

        # Image normalization constants
        self.mean = np.float32(np.array([0.485, 0.456, 0.406]))
        self.std = np.float32(np.array([0.229, 0.224, 0.225]))
        self.mean = self.mean / self.std
        self.std = self.std * 255.0

        # PnP solving
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
            # Pupils and eyeball centers
            [0.25799, 0.27608, -0.16923],
            [-0.25799, 0.27608, -0.16923],
            [0.25799, 0.27608, -0.24967],
            [-0.25799, 0.27608, -0.24967],
        ], np.float32) * np.array([1.0, 1.0, 1.3])

        self.camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
        self.inverse_camera = np.linalg.inv(self.camera)
        self.dist_coeffs = np.zeros((4,1))

        self.frame_count = 0
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
        self.res = 224. if self.model_type != 0 else 112.
        self.res_i = int(self.res)
        self.no_gaze = no_gaze
        self.debug_gaze = False
        self.face_info = [FaceInfo(id, self) for id in range(max_faces)]
        self.fail_count = 0

    def detect_faces(self, frame):
        im = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)[:,:,::-1] / self.std - self.mean
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        outputs, maxpool = self.detection.run([], {'input': im})
        outputs = np.array(outputs)
        maxpool = np.array(maxpool)
        outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
        detections = np.flip(np.argsort(outputs[0,0].flatten()))
        results = []
        for det in detections[0:self.max_faces]:
            y, x = det // 56, det % 56
            c = outputs[0, 0, y, x]
            r = outputs[0, 1, y, x] * 112.
            x *= 4
            y *= 4
            r *= 1.0
            if c < self.threshold:
                break
            results.append((x - r, y - r, 2 * r, 2 * r * 1.0))
        results = np.array(results).astype(np.float32)
        if results.shape[0] > 0:
            results[:, [0,2]] *= frame.shape[1] / 224.
            results[:, [1,3]] *= frame.shape[0] / 224.
        return results

    def landmarks(self, tensor, crop_info):
        crop_x1, crop_y1, scale_x, scale_y, _ = crop_info
        avg_conf = 0
        lms = []
        for i in range(0, 66):
            m = int(tensor[i].argmax())
            x = m // 28
            y = m % 28
            conf = float(tensor[i][x,y])
            avg_conf = avg_conf + conf
            off_x = self.res * ((1. * logit(tensor[66 + i][x, y])) - 0.0)
            off_y = self.res * ((1. * logit(tensor[66 * 2 + i][x, y])) - 0.0)
            off_x = math.floor(off_x + 0.5)
            off_y = math.floor(off_y + 0.5)
            lm_x = crop_y1 + scale_y * (self.res * (float(x) / 28.) + off_x)
            lm_y = crop_x1 + scale_x * (self.res * (float(y) / 28.) + off_y)
            lms.append((lm_x,lm_y,conf))
        avg_conf = avg_conf / 66.
        return (avg_conf, lms)

    def estimate_depth(self, face_info):
        lms = np.concatenate((face_info.lms, np.array([[face_info.eye_state[0][1], face_info.eye_state[0][2], face_info.eye_state[0][3]], [face_info.eye_state[1][1], face_info.eye_state[1][2], face_info.eye_state[1][3]]], np.float)), 0)

        image_pts = np.array(lms)[face_info.contour_pts, 0:2]

        success = False
        if not face_info.rotation is None:
            success, face_info.rotation, face_info.translation = cv2.solvePnP(face_info.contour, image_pts, self.camera, self.dist_coeffs, useExtrinsicGuess=True, rvec=np.transpose(face_info.rotation), tvec=np.transpose(face_info.translation), flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            rvec = np.array([0, 0, 0], np.float32)
            tvec = np.array([0, 0, 0], np.float32)
            success, face_info.rotation, face_info.translation = cv2.solvePnP(face_info.contour, image_pts, self.camera, self.dist_coeffs, useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation = face_info.rotation
        translation = face_info.translation

        pts_3d = np.zeros((70,3), np.float32)
        if not success:
            face_info.rotation = np.array([0.0, 0.0, 0.0], np.float32)
            face_info.translation = np.array([0.0, 0.0, 0.0], np.float32)
            return False, np.zeros(4), np.zeros(3), 99999., pts_3d, lms
        else:
            face_info.rotation = np.transpose(face_info.rotation)
            face_info.translation = np.transpose(face_info.translation)

        rmat, _ = cv2.Rodrigues(rotation)
        inverse_rotation = np.linalg.inv(rmat)
        pnp_error = 0.0
        for i, pt in enumerate(face_info.face_3d):
            if i == 68:
                # Right eyeball
                # Eyeballs have an average diameter of 12.5mm and and the distance between eye corners is 30-35mm, so a conversion factor of 0.385 can be applied
                eye_center = (pts_3d[36] + pts_3d[39]) / 2.0
                d_corner = np.linalg.norm(pts_3d[36] - pts_3d[39])
                depth = 0.385 * d_corner
                pt_3d = np.array([eye_center[0], eye_center[1], eye_center[2] - depth])
                pts_3d[i] = pt_3d
                continue
            if i == 69:
                # Left eyeball
                eye_center = (pts_3d[42] + pts_3d[45]) / 2.0
                d_corner = np.linalg.norm(pts_3d[42] - pts_3d[45])
                depth = 0.385 * d_corner
                pt_3d = np.array([eye_center[0], eye_center[1], eye_center[2] - depth])
                pts_3d[i] = pt_3d
                continue
            if i == 66:
                d1 = np.linalg.norm(lms[i,0:2] - lms[36,0:2])
                d2 = np.linalg.norm(lms[i,0:2] - lms[39,0:2])
                d = d1 + d2
                pt = (pts_3d[36] * d1 + pts_3d[39] * d2) / d
            if i == 67:
                d1 = np.linalg.norm(lms[i,0:2] - lms[42,0:2])
                d2 = np.linalg.norm(lms[i,0:2] - lms[45,0:2])
                d = d1 + d2
                pt = (pts_3d[42] * d1 + pts_3d[45] * d2) / d
            reference = rmat.dot(pt)
            reference = reference + face_info.translation
            reference = self.camera.dot(reference)
            depth = reference[2]
            if i < 17 or i == 30:
                reference = reference / depth
                e1 = lms[i][0] - reference[0]
                e2 = lms[i][1] - reference[1]
                pnp_error += e1*e1 + e2*e2
            pt_3d = np.array([lms[i][0] * depth, lms[i][1] * depth, depth], np.float32)
            pt_3d = self.inverse_camera.dot(pt_3d)
            pt_3d = pt_3d - face_info.translation
            pt_3d = inverse_rotation.dot(pt_3d)
            pts_3d[i,:] = pt_3d[:]

        pnp_error = np.sqrt(pnp_error / (2.0 * image_pts.shape[0]))
        if pnp_error > 300:
            self.fail_count += 1
            if self.fail_count > 5:
                # Something went wrong with adjusting the 3D model
                if not self.silent:
                    print(f"Detected anomaly when 3D fitting face {face_info.id}. Resetting.")
                face_info.face_3d = copy.copy(self.face_3d)
                face_info.rotation = None
                face_info.translation = np.array([0.0, 0.0, 0.0], np.float32)
                face_info.update_counts = np.zeros((66,2))
                face_info.update_contour()
        else:
            self.fail_count = 0

        euler = cv2.RQDecomp3x3(rmat)[0]
        return True, matrix_to_quaternion(rmat), euler, pnp_error, pts_3d, lms

    def preprocess(self, im, crop):
        x1, y1, x2, y2 = crop
        im = np.float32(im[y1:y2, x1:x2,::-1]) # Crop and BGR to RGB
        im = cv2.resize(im, (self.res_i, self.res_i), interpolation=cv2.INTER_LINEAR) / self.std - self.mean
        #im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0
        #im = (im - mean) / std
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        if self.model_type == 0:
            im = im.mean(1)
            im = np.expand_dims(im, 1)
        return im

    def equalize(self, im):
        im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])
        return cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)

    def corners_to_eye(self, corners, w, h, flip):
        ((cx1, cy1), (cx2, cy2)) = corners
        c1 = np.array([cx1, cy1])
        c2 = np.array([cx2, cy2])
        c2, a = compensate(c1, c2)
        center = (c1 + c2) / 2.0
        radius = np.linalg.norm(c1 - c2) / 2.0
        radius = np.array([radius * 1.4, radius * 1.2])
        upper_left = clamp_to_im(center - radius, w, h)
        lower_right = clamp_to_im(center + radius, w, h)
        return upper_left, lower_right, center, radius, c1, a

    def prepare_eye(self, frame, full_frame, lms, flip):
        outer_pt = tuple(lms[0])
        inner_pt = tuple(lms[1])
        h, w, _ = frame.shape
        (x1, y1), (x2, y2), center, radius, reference, a = self.corners_to_eye((outer_pt, inner_pt), w, h, flip)
        im = rotate_image(frame[:, :, ::], a, reference)
        im = im[int(y1):int(y2), int(x1):int(x2),:]
        if np.prod(im.shape) < 1:
            return None, None, None, None, None, None
        if flip:
            im = cv2.flip(im, 1)
        scale = np.array([(x2 - x1), (y2 - y1)]) / 32.
        im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_LINEAR)
        #im = self.equalize(im)
        if self.debug_gaze:
            if not flip:
                full_frame[0:32, 0:32] = im
            else:
                full_frame[0:32, 32:64] = im
        im = im.astype(np.float32)[:,:,::-1] / self.std - self.mean
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,2,1))
        return im, x1, y1, scale, reference, a

    def extract_face(self, frame, lms):
        lms = np.array(lms)[:,0:2][:,::-1]
        x1, y1 = tuple(lms.min(0))
        x2, y2 = tuple(lms.max(0))
        radius_x = 1.2 * (x2 - x1) / 2.0
        radius_y = 1.2 * (y2 - y1) / 2.0
        radius = np.array((radius_x, radius_y))
        center = (np.array((x1, y1)) + np.array((x2, y2))) / 2.0
        w, h, _ = frame.shape
        x1, y1 = clamp_to_im(center - radius, h, w)
        x2, y2 = clamp_to_im(center + radius + 1, h, w)
        offset = np.array((x1, y1))
        lms = (lms[:, 0:2] - offset).astype(np.int)
        frame = frame[y1:y2, x1:x2]
        return frame, lms, offset

    def get_eye_state(self, frame, lms):
        if self.no_gaze:
            return [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        lms = np.array(lms)
        e_x = [0,0]
        e_y = [0,0]
        scale = [0,0]
        reference = [None, None]
        angles = [0, 0]
        face_frame, lms, offset = self.extract_face(frame, lms)
        (right_eye, e_x[0], e_y[0], scale[0], reference[0], angles[0]) = self.prepare_eye(face_frame, frame, np.array([lms[36,0:2], lms[39,0:2]]), False)
        (left_eye, e_x[1], e_y[1], scale[1], reference[1], angles[1]) = self.prepare_eye(face_frame, frame, np.array([lms[42,0:2], lms[45,0:2]]), True)
        if right_eye is None or left_eye is None:
            return [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        both_eyes = np.concatenate((right_eye, left_eye))
        results = self.gaze_model.run([], {self.input_name: both_eyes})
        open = [0, 0]
        open[0] = 1#results[1][0].argmax()
        open[1] = 1#results[1][1].argmax()
        results = np.array(results[0])

        eye_state = []
        for i in range(2):
            m = int(results[i][0].argmax())
            x = m // 8
            y = m % 8
            conf = float(results[i][0][x,y])

            off_x = 32.0 * logit(results[i][1][x, y], 8.0)
            off_y = 32.0 * logit(results[i][2][x, y], 8.0)
            if i == 1:
                eye_x = 32.0 * float(x) / 8.0 + off_x
            else:
                eye_x = 32.0 * float(x) / 8.0 + off_x
            eye_y = 32.0 * float(y) / 8.0 + off_y

            if self.debug_gaze:
                if i == 0:
                    frame[int(eye_y), int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), int(eye_x+1)] = (0, 0, 255)
                    frame[int(eye_y), int(eye_x+1)] = (0, 0, 255)
                else:
                    frame[int(eye_y), 32+int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), 32+int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), 32+int(eye_x+1)] = (0, 0, 255)
                    frame[int(eye_y), 32+int(eye_x+1)] = (0, 0, 255)

            if i == 0:
                eye_x = e_x[i] + scale[i][0] * eye_x
            else:
                eye_x = e_x[i] + scale[i][0] * (32. - eye_x)
            eye_y = e_y[i] + scale[i][1] * eye_y
            eye_x, eye_y = rotate(reference[i], (eye_x, eye_y), -angles[i])

            eye_x = eye_x + offset[0]
            eye_y = eye_y + offset[1]
            eye_state.append([open[i], eye_y, eye_x, conf])

        return eye_state

    def assign_face_info(self, results):
        result_coords = []
        adjusted_results = []
        for conf, (lms, eye_state), conf_adjust in results:
            adjusted_results.append((conf - conf_adjust, (lms, eye_state)))
            result_coords.append(np.array(lms)[:, 0:2].mean(0))
        results = adjusted_results
        candidates = [[]] * self.max_faces
        max_dist = 2 * np.linalg.norm(np.array([self.width, self.height]))
        for i, face_info in enumerate(self.face_info):
            for j, coord in enumerate(result_coords):
                if face_info.coord is None:
                    candidates[i].append((max_dist, i, j))
                else:
                    candidates[i].append((np.linalg.norm(face_info.coord - coord), i, j))
        for i, candidate in enumerate(candidates):
            candidates[i] = sorted(candidate)
        found = 0
        target = len(results)
        used_results = {}
        used_faces = {}
        while found < target:
            min_list = min(candidates)
            candidate = min_list.pop(0)
            face_idx = candidate[1]
            result_idx = candidate[2]
            if not result_idx in used_results and not face_idx in used_faces:
                self.face_info[face_idx].update(results[result_idx], result_coords[result_idx], self.frame_count)
                min_list.clear()
                used_results[result_idx] = True
                used_faces[face_idx] = True
                found += 1
            if len(min_list) == 0:
                min_list.append((2 * max_dist, face_idx, result_idx))
        for face_info in self.face_info:
            if face_info.frame_count != self.frame_count:
                face_info.update(None, None, self.frame_count)

    def predict(self, frame, additional_faces=[]):
        self.frame_count += 1
        start = time.perf_counter()
        im = frame

        duration_fd = 0.0
        duration_pp = 0.0
        duration_model = 0.0
        duration_pnp = 0.0

        new_faces = []
        new_faces.extend(self.faces)
        bonus_cutoff = len(self.faces)
        new_faces.extend(additional_faces)
        self.wait_count += 1
        if self.detected == 0:
            start_fd = time.perf_counter()
            if self.use_retinaface > 0:
                retinaface_detections = self.retinaface.detect_retina(frame)
                new_faces.extend(retinaface_detections)
            else:
                new_faces.extend(self.detect_faces(frame))
            duration_fd = 1000 * (time.perf_counter() - start_fd)
            self.wait_count = 0
        elif self.detected < self.max_faces:
            if self.use_retinaface > 0:
                new_faces.extend(self.retinaface_scan.get_results())
            if self.wait_count >= self.scan_every:
                if self.use_retinaface > 0:
                    self.retinaface_scan.background_detect(frame)
                else:
                    start_fd = time.perf_counter()
                    new_faces.extend(self.detect_faces(frame))
                    duration_fd = 1000 * (time.perf_counter() - start_fd)
                    self.wait_count = 0
        else:
            self.wait_count = 0

        if len(new_faces) < 1:
            duration = (time.perf_counter() - start) * 1000
            if not self.silent:
                print(f"Took {duration:.2f}ms")
            return []

        crops = []
        crop_info = []
        num_crops = 0
        for j, (x,y,w,h) in enumerate(new_faces):
            crop_x1 = x - int(w * 0.1)
            crop_y1 = y - int(h * 0.125)
            crop_x2 = x + w + int(w * 0.1)
            crop_y2 = y + h + int(h * 0.125)

            crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), self.width, self.height)
            crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), self.width, self.height)

            scale_x = float(crop_x2 - crop_x1) / self.res
            scale_y = float(crop_y2 - crop_y1) / self.res

            if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                continue

            start_pp = time.perf_counter()
            crop = self.preprocess(im, (crop_x1, crop_y1, crop_x2, crop_y2))
            duration_pp += 1000 * (time.perf_counter() - start_pp)
            crops.append(crop)
            crop_info.append((crop_x1, crop_y1, scale_x, scale_y, 0.0 if j >= bonus_cutoff else 0.1))
            num_crops += 1

        groups = group_rects(new_faces)

        start_model = time.perf_counter()
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
            eye_state = self.get_eye_state(frame, lms)
            if conf < self.threshold:
                continue;
            group_id = groups[str(new_faces[i])][0]
            if not group_id in best_results:
                best_results[group_id] = [-1, [], 0]
            if conf > self.threshold and best_results[group_id][0] < conf + crop[4]:
                best_results[group_id][0] = conf + crop[4]
                best_results[group_id][1] = (lms, eye_state)
                best_results[group_id][2] = crop[4]

        sorted_results = sorted(best_results.values(), key=lambda x: x[0], reverse=True)[:self.max_faces]
        self.assign_face_info(sorted_results)
        duration_model = 1000 * (time.perf_counter() - start_model)

        results = []
        detected = []
        for face_info in self.face_info:
            if face_info.alive and face_info.conf > self.threshold:
                start_pnp = time.perf_counter()
                face_info.success, face_info.quaternion, face_info.euler, face_info.pnp_error, face_info.pts_3d, face_info.lms = self.estimate_depth(face_info)
                face_info.adjust_3d()
                duration_pnp += 1000 * (time.perf_counter() - start_pnp)
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for (x,y,c) in face_info.lms[0:66]:
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
                bbox = (min_y, min_x, max_y - min_y, max_x - min_x)
                detected.append(bbox)
                face_info.bbox = bbox
                #face_info.eye_blink = (eye_state[0][0], eye_state[1][0])
                results.append(face_info)

        if len(detected) > 0:
            self.detected = len(detected)
            self.faces = detected
            self.discard = 0
        else:
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

        duration = (time.perf_counter() - start) * 1000
        if not self.silent:
            print(f"Took {duration:.2f}ms (detect: {duration_fd:.2f}ms, crop: {duration_pp:.2f}, track: {duration_model:.2f}ms, 3D points: {duration_pnp:.2f}ms)")

        results = sorted(results, key=lambda x: x.id)

        return results
