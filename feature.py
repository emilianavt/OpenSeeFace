import os
import numpy as np
import math
import cv2

#got tired of trying to remember which order I had to do the nested min and max
def clamp (value, minimum, maxium):
    return max(min(value,maxium),minimum)

def rotate(origin, point, a):
    x, y = point - origin
    cosa = math.cos(-a)
    sina = math.sin(-a)
    qx = origin[0] + cosa * x - sina * y
    qy = origin[1] + sina * x + cosa * y
    return qx, qy

#I redid a lot of this so it worked well for me
#idk if it'll work well for other people
class Feature():
    def __init__(self, alpha=0.1, decay=0.0005, curve=1, scaleType = 1):
        self.min = None
        self.max = None
        self.alpha = alpha
        self.decay = decay
        self.last = 0.0
        self.curve = curve
        self.scaleType = scaleType

    def update(self, x):
        new = self.update_state(x)
        self.last = self.last * self.alpha + new * (1 - self.alpha)
        return self.last

    def update_state(self, x):
        if self.min is None or self.max is None:
            self.min = x - 0.00001
            self.max = x + 0.00001
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        center = (self.min+self.max)/2

        #The Min and Max decay slightly toward each other every frame allowing the range to change dynamically
        self.min = (self.min * (1 - self.decay)) + (self.max * self.decay)
        self.max = (self.max * (1 - self.decay)) + (self.min * self.decay)

        if self.scaleType == 1:
            #Returns a value between -1 and 1 in relation to the maximum range
            if x < center:
                return -pow(clamp((x - center) / (self.min - center), 0, 1), self.curve)
            elif x > center:
                return pow(clamp((x - center) / (self.max - center), 0, 1), self.curve)
            return 0
        if self.scaleType == 2:
            #Returns a value between 0 and 1 in relation to the maximum range
                return pow(clamp((x - self.min) / (self.max - self.min), 0, 1), self.curve)
        return 0

class FeatureExtractor():
    def __init__(self):
        self.eye_l = Feature(scaleType = 2, curve = 0.5)
        self.eye_r = Feature(scaleType = 2, curve = 0.5)
        self.eyebrow_updown_l = Feature( curve = 1.5)
        self.eyebrow_updown_r = Feature( curve = 1.5)
        self.mouth_corner_updown_l = Feature(curve = 2, decay=0.00005  )
        self.mouth_corner_updown_r = Feature(curve = 2, decay=0.00005 )
        self.mouth_open = Feature(scaleType = 2, curve = 1.5)

    def align_points(self, a, b, pts): #runs 6 times per frame

        alpha = (math.atan2(*(b - a)[::-1]) % (math.pi *2))

        if alpha >= (math.pi/2):
            alpha = - (alpha - math.pi)
        if alpha <= -(math.pi/2):
            alpha = - (alpha + math.pi)
        aligned_pts = []
        for pt in pts:
            aligned_pts.append(rotate(a, pt, alpha))
        return (aligned_pts)

    def update(self, pts):
        features = {}
        norm_distance_y = pts[27, 1]-pts[8, 1]

        f_pts = self.align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
        f = (f_pts[0][1]/2 + f_pts[1][1]/2) - (f_pts[2][1]/2 + f_pts[3][1]/2)
        features["eye_l"] =  self.eye_l.update(f)
        f_pts = self.align_points(pts[36], pts[39], pts[[37, 38, 41, 40]])
        f = (f_pts[0][1]/2 + f_pts[1][1]/2) - (f_pts[2][1]/2 + f_pts[3][1]/2)
        features["eye_r"] = self.eye_r.update(f)

        f = ((pts[22][1]/2 + pts[26][1]/2) - (pts[42][1]/2 + pts[45][1]/2))/norm_distance_y
        features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f)
        f = ((pts[17][1]/2 + pts[21][1]/2) - (pts[36][1]/2 + pts[39][1]/2))/norm_distance_y
        features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f)

        f = ( pts[50][1] - pts[55][1]) * (norm_distance_y)
        features["mouth_open"] = self.mouth_open.update(f)*0.66

        #mouth corners are calculated together so they are assigned the same value
        f = (((pts[51][1] + pts[55][1])/4) - ((pts[58][1]+pts[62][1])/2))/ norm_distance_y
        features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f) * (1- 0.3 *features["mouth_open"]) * 0.66 - 0.1
        features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f) * (1- 0.3 * features["mouth_open"]) *0.66 - 0.1

        #i removed some features VTS didn't seem to use
        #the VTS library was handling them not being set anyway, so I just let that deal with the fact they're not here
        return features
