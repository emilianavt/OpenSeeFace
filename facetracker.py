import os
os.environ["OMP_NUM_THREADS"] = str(1)
import argparse
import traceback
import threading
import queue
import time
from tracker import Tracker
import multiprocessing
import webcam
import vts
import preview

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=480)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default=0)
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=0.6)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized.", default=3, choices=[0, 1, 2, 3, 4])
parser.add_argument("--preview", type=int, help="Preview the frames sent to the tracker", default=0)
parser.add_argument("--feature-type", type=int, help="Sets which version of feature extraction is used. 0 is my new version that works well for me and allows for some customization, 1 is EmilianaVT's version aka, normal OpenSeeFace operation", default=0, choices=[0, 1])
parser.add_argument("--numpy-threads", type=int, help="Numer of threads Numpy can use, doesn't seem to effect much", default=1)
parser.add_argument("-T","--threads", type=int, help="Numer of threads used for landmark detection. Default is 1 (~15ms per frame on my computer), 2 gets slightly faster frames (~10ms on my computer), more than 2 doesn't seem to help much", default=1)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking", default=0)
parser.add_argument("--low-latency", type=int, help="Low latency mode. Lowers latency, at the cost of inconsistent timings", default = 0)
parser.add_argument("--target-brightness", type=float, help="range 0.25-0.75, Target brightness of the brightness adjustment. Defaults to 0.55", default = 0.55)




args = parser.parse_args()

#processing args
target_ip = args.ip
target_port = args.port
fps = args.fps
height = args.height
width = args.width
previewFlag = (args.preview == 1)
mirror_input = args.mirror_input
featureType = args.feature_type
visualizeFlag = (args.visualize == 1)
lowLatency = (args.low_latency == 1)
threads = args.threads
targetBrightness = min(max(args.target_brightness, 0.25), 0.75)
silent = (args.silent == 1)


if lowLatency:
    target_duration = 0.75 / fps
    frameQueueSize = 1
elif height > 480:
    target_duration = 1 / fps
    frameQueueSize = 2
else:
    frameQueueSize = 1
    target_duration = 1 / fps


frameQueue = queue.Queue(maxsize=frameQueueSize)
faceQueue = queue.Queue(maxsize=1)
packetQueue = queue.Queue()

#I want to decouple the requests so they don't block anything else if they're slow
VTS = vts.VTS(target_ip, target_port, silent, height, width, packetQueue)
packetSenderThread = threading.Thread(target = VTS.packetSender, args = [],)
packetSenderThread.daemon = True
packetSenderThread.start()


webcamThread = threading.Thread(target=webcam.startProcess,args = (frameQueue, faceQueue, fps, targetBrightness, width, height,  mirror_input, ))
webcamThread.daemon = True
webcamThread.start()

if previewFlag or visualizeFlag:
    previewFrameQueue = queue.Queue(maxsize=1)
    previewThread = threading.Thread(target=preview.startProcess, args = (previewFrameQueue, target_duration,))
    previewThread.daemon = True
    previewThread.start()

frame_count = 0
peak_frame_time=0.0
total_active_time = 0.0
total_run_time = 0.0
frame_start = 0.0
peak_time_between = 0.0
peak_camera_latency = 0.0
total_camera_latency = 0.0
sleepTimer = 0.0

failures = 0
peakTotalLatency = 0.0
totalTotalLatency = 0.0

tracker = Tracker(width, height, featureType, threads, threshold=args.threshold, silent=silent, model_type=args.model, detection_threshold=args.detection_threshold)

#don't start until the webcam is ready, then give it a little more time to fill it's buffer
frameQueue.get()
time.sleep(0.1)

face = None
totalFrameLatency = 0
try:
    while True:
        #clearing these so there's no stale data to confuse the checks
        packet = None
        frame_start = time.perf_counter()
        frame_count += 1
        frame, camera_latency, totalFrameLatency = frameQueue.get()
        frame_get = time.perf_counter()
        #If I don't wait a few frames to start tracking I get wild peak frame times, like 500ms
        if frame_count > 5:
            peak_camera_latency = max(camera_latency, peak_camera_latency)
            total_camera_latency+= camera_latency

        if previewFlag:
            #there's no reason to ever wait on this process or give it much of a queue
            if previewFrameQueue.qsize() < 1:
                previewFrameQueue.put(frame)

        faceInfo, face  = tracker.predict(frame)

        if faceInfo is not None:
            packet = VTS.preparePacket(faceInfo)
            if faceQueue.qsize() < 1:
                faceQueue.put(face)
            if visualizeFlag:
                preview.visualize(frame, faceInfo, previewFrameQueue, face_center, face_radius)

        frameTime = time.perf_counter() - frame_get
        total_active_time += frameTime
        peak_frame_time = max(peak_frame_time, frameTime)

        duration = time.perf_counter() - frame_start
        time.sleep(max(0, target_duration - duration))

        peak_time_between = max(peak_time_between, time.perf_counter() -frame_start)
        total_run_time += time.perf_counter() -frame_start

        #If we don't have something to send to Vtube Studio we don't
        if(packet is not None):
            packetQueue.put(packet)
        else:
            print("No data sent to VTS")
        if frame_count > 5:
            peakTotalLatency = max(peakTotalLatency, (time.perf_counter() - totalFrameLatency))
            totalTotalLatency += time.perf_counter() - totalFrameLatency

except KeyboardInterrupt:
    if not silent:
        print("Quitting")

#printing statistics on close
#it makes identifying problems easier

print(f"Peak latency: {(peakTotalLatency*1000):.3f}ms")
print(f"Average latency: {(totalTotalLatency*1000/(frame_count-5)):.3f}ms")
#480p generally results in camera latencies between 0.5ms and 2.0ms
print(f"Peak camera latency: {(peak_camera_latency*1000):.3f}ms")
print(f"Average camera latency: {(total_camera_latency*1000/(frame_count-5)):.3f}ms")
#The longest time a full cycle of the main loop takes, including sleep
#indicates issues when higher than average time between frames by a significant amount
print(f"Peak time between frames: {(peak_time_between*1000):.3f}ms")
#at 30fps this should be a hair above 33.333ms
print(f"Average time between frames: {(total_run_time*1000/(frame_count)):.3f}ms")
#These measure how long the main loop takes *not including sleep*
#tends to be slightly longer than the average time given by the tracking status by 1ms to 2ms
#because it includes detection frames and all the other minor operations that need to happen every frame
print(f"Peak frame time: {(peak_frame_time*1000):.3f}ms")
print(f"Average frame time: { ((total_active_time)*1000/(frame_count)):.3f}ms")
print(f"Run time (seconds): {total_run_time:.2f} s\nFrames: {frame_count}")

os._exit(0)
