import os
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to a number higher than 1 to output only the names", default=0)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.4)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking or to 2 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select between the two models with 0 being the original model that might be overall more accurate, 1 being a model that might be able to fit eye and eyebrow shapes more accurately and 2 being a much faster, but lower accuracy model", default=1, choices=[0, 1, 2])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--high-quality-3d", type=int, help="When set to 1, more nose points are used when estimating the face pose", default=1)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "1"

if os.name == 'nt' and args.list_cameras > 0:
    import escapi
    escapi.init()
    camera_count = escapi.count_capture_devices()
    if args.list_cameras == 1:
        print("Available cameras:")
    for i in range(camera_count):
        camera_name = escapi.device_name(i).decode()
        if args.list_cameras == 1:
            print(f"{i}: {camera_name}")
        else:
            print(camera_name)
    sys.exit(0)

import numpy as np
import time
import cv2
import socket
import struct
from input_reader import InputReader, list_cameras
from tracker import Tracker

target_ip = args.ip
target_port = args.port

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")

fps = 0
if os.name == 'nt':
    fps = args.fps
input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps)

log = None
out = None
first = True
height = 0
width = 0
tracker = None
sock = None
tracking_time = 0.0
tracking_frames = 0
frame_count = 0

if args.log_data != "":
    log = open(args.log_data, "w")
    log.write("Frame,Time,Face,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
    for i in range(66):
        log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
    for i in range(66):
        log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
    log.write("\r\n")
    log.flush()

try:
    while input_reader.is_open():
        if not input_reader.is_ready():
            time.sleep(0.001)
            continue

        ret, frame = input_reader.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, pnp_quality=args.high_quality_3d, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 else True)
            if not args.video_out is None:
                out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('M','J','P','G'), 24, (width,height))

        inference_start = time.perf_counter()
        faces = tracker.predict(frame)
        if len(faces) > 0:
            tracking_time += (time.perf_counter() - inference_start) / len(faces)
            tracking_frames += 1
        packet = bytearray()
        detected = False
        for face_num, (conf, lms, success_3d, pnp_error, quaternion, euler, rotation, translation, pts_3d, (right_open, left_open), (face_y, face_x, face_height, face_width)) in enumerate(faces):
            left_state = "O" if left_open > 0.5 else "-"
            right_state = "O" if right_open > 0.5 else "-"
            if args.silent == 0:
                print(f"Confidence: {conf:.4f} / 3D fitting error: {pnp_error:.4f} / Eyes: {left_state}, {right_state}")
            detected = True
            if not success_3d:
                pts_3d = np.zeros((70, 3), np.float32)
            packet.extend(bytearray(struct.pack("d", now)))
            packet.extend(bytearray(struct.pack("f", right_open)))
            packet.extend(bytearray(struct.pack("f", left_open)))
            packet.extend(bytearray(struct.pack("B", 1 if success_3d else 0)))
            packet.extend(bytearray(struct.pack("f", pnp_error)))
            packet.extend(bytearray(struct.pack("f", quaternion[0])))
            packet.extend(bytearray(struct.pack("f", quaternion[1])))
            packet.extend(bytearray(struct.pack("f", quaternion[2])))
            packet.extend(bytearray(struct.pack("f", quaternion[3])))
            packet.extend(bytearray(struct.pack("f", euler[0])))
            packet.extend(bytearray(struct.pack("f", euler[1])))
            packet.extend(bytearray(struct.pack("f", euler[2])))
            packet.extend(bytearray(struct.pack("f", translation[0])))
            packet.extend(bytearray(struct.pack("f", translation[1])))
            packet.extend(bytearray(struct.pack("f", translation[2])))
            if not log is None:
                log.write(f"{frame_count},{now},{face_num},{right_open},{left_open},{conf},{success_3d},{pnp_error},{quaternion[0]},{quaternion[1]},{quaternion[2]},{quaternion[3]},{rotation[0][0]},{rotation[0][1]},{rotation[0][2]},{translation[0][0]},{translation[0][1]},{translation[0][2]}")
            for (x,y,c) in lms:
                packet.extend(bytearray(struct.pack("f", c)))
            for pt_num, (x,y,c) in enumerate(lms):
                packet.extend(bytearray(struct.pack("f", y)))
                packet.extend(bytearray(struct.pack("f", x)))
                if not log is None:
                    log.write(f",{y},{x},{c}")
                if pt_num == 66:# and right_open < 0.5:
                    continue
                if pt_num == 67:# and left_open < 0.5:
                    continue
                if args.visualize != 0 or not out is None:
                    if args.visualize > 1:
                        frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 0, 255)
                    x += 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 0, 255)
                    y += 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 0, 255)
                    x -= 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 0, 255)
            if args.pnp_points != 0 and (args.visualize != 0 or not out is None):
                projected = cv2.projectPoints(tracker.contour, rotation, translation, tracker.camera, tracker.dist_coeffs)
                for [(x,y)] in projected[0]:
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 255, 255)
                    x += 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 255, 255)
                    y += 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 255, 255)
                    x -= 1
                    if not (x < 0 or y < 0 or x >= height or y >= width):
                        frame[int(x), int(y)] = (0, 255, 255)
            for (x,y,z) in pts_3d:
                packet.extend(bytearray(struct.pack("f", x)))
                packet.extend(bytearray(struct.pack("f", -y)))
                packet.extend(bytearray(struct.pack("f", -z)))
                if not log is None:
                    log.write(f",{x},{-y},{-z}")
            if not log is None:
                log.write("\r\n")
                log.flush()

        if detected and len(faces) < 40:
            sock.sendto(packet, (target_ip, target_port))

        if not out is None:
            out.write(frame)

        if args.visualize != 0:
            cv2.imshow('OpenSeeFace Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")

input_reader.close()
if not out is None:
    out.release()
cv2.destroyAllWindows()

if args.silent == 0 and tracking_frames > 0:
    tracking_time = 1000 * tracking_time / tracking_frames
    print(f"Average tracking time per detected face: {tracking_time:.2f} ms")
