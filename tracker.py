import numpy as np
import cv2
import time
import facedetection
import eyes
import landmarks
import face
import featureExtractor


#this file is so much smaller than it used to be, lol
#I moved most of the functionality into separate files so they're easier to work with
def extract_face( frame, lms):
    x1, y1 = lms.min(0)
    x2, y2 = lms.max(0)
    radius = np.array([(x2 - x1), (y2 - y1)])*0.6
    center = (np.array((x1, y1)) + np.array((x2, y2))) / 2.0
    w, h, _ = frame.shape
    x1, y1 = clamp_to_im(center - radius, h, w)
    x2, y2 = clamp_to_im(center + radius + 1, h, w)
    offset = np.array((x1, y1))
    lms = (lms[:, 0:2] - offset).astype(int)
    frame = frame[y1:y2, x1:x2]
    return frame, lms, offset, center, radius

def clamp_to_im(pt, w, h): #8 times per frame, but that only accounts for 0.005ms
    x=max(pt[0],0)
    y=max(pt[1],0)
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x),int(y+1))

class Tracker():
    def __init__(self, width, height, messageQueue, model_type=3, detection_threshold=0.6, threshold=0.6, silent=False):

        self.FaceDetector = facedetection.FaceDetector(detection_threshold = detection_threshold)
        self.EyeTracker = eyes.EyeTracker()
        self.Landmarks = landmarks.Landmarks(width, height, model_type, threshold)
        self.FeatureExtractor = featureExtractor.FeatureExtractor()

        # Image normalization constants
        self.mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
        self.std = np.float32(np.array([0.0171, 0.0175, 0.0174]))

        self.camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
        self.inverse_camera = np.linalg.inv(self.camera)
        self.width = width
        self.height = height

        self.threshold = threshold

        self.faces = []
        self.face_info = [face.FaceInfo(0, self)]
        self.messageQueue = messageQueue

    def preprocess(self, im):
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC) * self.std + self.mean
        im = np.expand_dims(im, 0)
        return np.transpose(im, (0,3,1,2))

    def cropFace(self,face, im):
        duration_pp = 0.0
        x,y,w,h = face

        wint = int(w * 0.1)
        hint = int(h * 0.125)
        crop_x1 = x - wint
        crop_y1 = y - hint
        crop_x2 = x + w + wint
        crop_y2 = y + h + hint

        crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), self.width, self.height)
        crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), self.width, self.height)
        cropx = (crop_x2 - crop_x1)
        cropy = (crop_y2 - crop_y1)

        if cropx < 64 or cropy < 64:
            return (None, [], duration_pp )

        start_pp = time.perf_counter()
        crop = self.preprocess(im[crop_y1:crop_y2, crop_x1:crop_x2])
        duration_pp += 1000 * (time.perf_counter() - start_pp)

        scale_x = cropx / 224.
        scale_y = cropy / 224.
        crop_info = (crop_x1, crop_y1, scale_x, scale_y)
        return (crop, crop_info, duration_pp )

    def early_exit(self, reason, start):
        self.messageQueue.put(reason)
        self.faces = []
        duration = (time.perf_counter() - start) * 1000
        self.messageQueue.put(f"Took {duration:.2f}ms")
        return None, None, None

    def predict(self, frame, face = None):
        start = time.perf_counter()
        duration_fd = 0.0
        duration_pp = 0.0
        duration_model = 0.0
        duration_pnp = 0.0

        new_faces = []
        new_faces.extend(self.faces)
        if face is not None:
            new_faces.append(face)

        if len(new_faces) < 1:
            start_fd = time.perf_counter()
            new_faces.extend(self.FaceDetector.detect_faces(frame))
            duration_fd = 1000 * (time.perf_counter() - start_fd)
        if len(new_faces) < 1:
            return self.early_exit("No faces found", start)

        crop, crop_info, duration_pp = self.cropFace(new_faces[0], frame)

        #Early exit if crop fails, If the crop fails there's nothing to track
        if  crop is None:
            return self.early_exit("No valid crops", start)

        start_model = time.perf_counter()
        #self.Landmarks

        conf, lms = self.Landmarks.run(crop , crop_info )
        #Early exit if below confidence threshold
        if conf < self.threshold:
            return self.early_exit("Confidence below threshold", start)

        #lms is short for landmarks
        face_frame, face_lms, face_offset,face_center, face_radius = extract_face(frame, np.array(lms)[:,0:2][:,::-1])
        eye_state = self.EyeTracker.get_eye_state(face_frame, face_lms, face_offset)

        self.face_info[0].update((conf, (lms, eye_state)), np.array(lms)[:, 0:2].mean(0))

        duration_model = 1000 * (time.perf_counter() - start_model)
        start_pnp = time.perf_counter()
        face_info = self.face_info[0]

        if face_info.alive:
            face_info.success, face_info.quaternion, face_info.euler, face_info.pnp_error, face_info.pts_3d, face_info.lms = self.Landmarks.estimate_depth(face_info)

            if face_info.success:
                face_info.adjust_3d()
                lms = face_info.lms[:, 0:2]
                x1, y1 = lms[0:66].min(0)
                x2, y2 = lms[0:66].max(0)
                self.faces = [[y1, x1, y2 - y1, x2 - x1]]
                duration_pnp += 1000 * (time.perf_counter() - start_pnp)
                duration = (time.perf_counter() - start) * 1000
                self.messageQueue.put(f"Took {duration:.2f}ms (detect: {duration_fd:.2f}ms, crop: {duration_pp:.2f}ms, track: {duration_model:.2f}ms, 3D points: {duration_pnp:.2f}ms)")
                return face_info, face_center, face_radius

        #Combined multiple failures into one catch all exit
        return self.early_exit("Face info not valid", start)
