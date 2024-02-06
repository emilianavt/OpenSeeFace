import numpy as np
import time
import math
import emilianaFeature

def angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.arctan2(*(p2 - p1)[::-1])
    return (a % (2 * np.pi))

def rotate(origin, point, a):
    a = -a
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy

class FeatureExtractor():
    def __init__(self, max_feature_updates=0):
        self.eye_l = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.eye_r = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_l = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_r = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_l = emilianaFeature.Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_r = emilianaFeature.Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_l = emilianaFeature.Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_r = emilianaFeature.Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_l = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_r = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_l = emilianaFeature.Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_r = emilianaFeature.Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_open = emilianaFeature.Feature(max_feature_updates=max_feature_updates)
        self.mouth_wide = emilianaFeature.Feature(threshold=0.02, max_feature_updates=max_feature_updates)

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

    def update(self, pts, full=True):
        features = {}
        now = time.perf_counter()

        norm_distance_x = np.mean([pts[0, 0] - pts[16, 0], pts[1, 0] - pts[15, 0]])
        norm_distance_y = np.mean([pts[27, 1] - pts[28, 1], pts[28, 1] - pts[29, 1], pts[29, 1] - pts[30, 1]])

        a1, f_pts = self.align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
        f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        features["eye_l"] = self.eye_l.update(f, now)

        a2, f_pts = self.align_points(pts[36], pts[39], pts[[37, 38, 41, 40]])
        f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        features["eye_r"] = self.eye_r.update(f, now)

        if full:
            a3, _ = self.align_points(pts[0], pts[16], [])
            a4, _ = self.align_points(pts[31], pts[35], [])
            norm_angle = np.mean(list(map(np.rad2deg, [a1, a2, a3, a4])))

            a, f_pts = self.align_points(pts[22], pts[26], pts[[22, 23, 24, 25, 26]])
            features["eyebrow_steepness_l"] = self.eyebrow_steepness_l.update(-np.rad2deg(a) - norm_angle, now)
            f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
            features["eyebrow_quirk_l"] = self.eyebrow_quirk_l.update(f, now)

            a, f_pts = self.align_points(pts[17], pts[21], pts[[17, 18, 19, 20, 21]])
            features["eyebrow_steepness_r"] = self.eyebrow_steepness_r.update(np.rad2deg(a) - norm_angle, now)
            f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
            features["eyebrow_quirk_r"] = self.eyebrow_quirk_r.update(f, now)
        else:
            features["eyebrow_steepness_l"] = 0.
            features["eyebrow_steepness_r"] = 0.
            features["eyebrow_quirk_l"] = 0.
            features["eyebrow_quirk_r"] = 0.

        f = (np.mean([pts[22, 1], pts[26, 1]]) - pts[27, 1]) / norm_distance_y
        features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f, now)

        f = (np.mean([pts[17, 1], pts[21, 1]]) - pts[27, 1]) / norm_distance_y
        features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f, now)

        upper_mouth_line = np.mean([pts[49, 1], pts[50, 1], pts[51, 1]])
        center_line = np.mean([pts[50, 0], pts[60, 0], pts[27, 0], pts[30, 0], pts[64, 0], pts[55, 0]])

        f = (upper_mouth_line - pts[62, 1]) / norm_distance_y
        features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f, now)
        if full:
            f = abs(center_line - pts[62, 0]) / norm_distance_x
            features["mouth_corner_inout_l"] = self.mouth_corner_inout_l.update(f, now)
        else:
            features["mouth_corner_inout_l"] = 0.

        f = (upper_mouth_line - pts[58, 1]) / norm_distance_y
        features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f, now)
        if full:
            f = abs(center_line - pts[58, 0]) / norm_distance_x
            features["mouth_corner_inout_r"] = self.mouth_corner_inout_r.update(f, now)
        else:
            features["mouth_corner_inout_r"] = 0.

        f = abs(np.mean(pts[[59,60,61], 1], axis=0) - np.mean(pts[[63,64,65], 1], axis=0)) / norm_distance_y
        features["mouth_open"] = self.mouth_open.update(f, now)

        f = abs(pts[58, 0] - pts[62, 0]) / norm_distance_x
        features["mouth_wide"] = self.mouth_wide.update(f, now)

        return features
