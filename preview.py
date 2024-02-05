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

