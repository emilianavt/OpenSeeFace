import cv2
import multiprocessing

#this is it's own file just for ceanlieness

def startProcess(frameQueue, targetFrameTime):
    preview = Preview(frameQueue, targetFrameTime)
    preview.start()

class Preview():
    def __init__(self, frameQueue, targetFrameTime):
        self.frameQueue = frameQueue
        self.targetFrameTime = targetFrameTime

    def start(self):
        while True:
            #I had a timer on this, but tbh it's fine to just block it with the main thread
            frame = self.frameQueue.get()
            cv2.imshow("test",cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

def visualize(frame, face, previewFrameQueue, face_center, face_radius):

    y1 = int(face_center[1] - face_radius[1])
    y2 = int(face_center[1] + face_radius[1])
    x1 = int(face_center[0] - face_radius[0])
    x2 = int(face_center[0] + face_radius[0])
    bbox = (y1, x1, y2 - y1, x2 - x1)
    frame = cv2.putText(frame, str(face.id), (int(bbox[1]), int(bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255))
    frame = cv2.putText(frame, f"{face.conf:.4f}", (int(bbox[1] + 18), int(bbox[0] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    for pt_num, (x,y,c) in enumerate(face.lms):
        x = int(x + 0.5)
        y = int(y + 0.5)
        frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
        color = (0, 255, 0)
        if pt_num >= 66:
            color = (255, 255, 0)
        if not (x < 0 or y < 0 or x >= height or y >= width):
            cv2.circle(frame, (y, x), 1, color, -1)
    previewFrameQueue.put(frame)
