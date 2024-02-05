import time
import socket
import struct

class VTS():
    def __init__(self, target_ip, target_port, silent, height, width, packetQueue, messageQueue):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_ip = target_ip
        self.target_port = target_port
        self.silent = silent
        self.features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]
        self.width = width
        self.height = height
        self.messageQueue = messageQueue
        self.packetQueue = packetQueue

    def packetSender(self):
        while True:
            packet = self.packetQueue.get()
            self.sendPacket(packet)

    def preparePacket(self, face):
        packet = bytearray()
        if face.eye_blink is None:
            face.eye_blink = [1, 1]
        if not self.silent:
            right_state = "O" if face.eye_blink[0] > 0.2 else "-"
            left_state = "O" if face.eye_blink[1] > 0.2 else "-"
            print(f"Confidence: {face.conf:.4f} / 3D fitting error: {face.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
        now = time.time()
        packet.extend(bytearray(struct.pack("d", now)))
        packet.extend(bytearray(struct.pack("i", face.id)))
        packet.extend(bytearray(struct.pack("f", self.width)))
        packet.extend(bytearray(struct.pack("f", self.height)))
        packet.extend(bytearray(struct.pack("f", face.eye_blink[0])))
        packet.extend(bytearray(struct.pack("f", face.eye_blink[1])))
        packet.extend(bytearray(struct.pack("B", 1)))
        packet.extend(bytearray(struct.pack("f", face.pnp_error)))
        packet.extend(bytearray(struct.pack("f", face.quaternion[0])))
        packet.extend(bytearray(struct.pack("f", face.quaternion[1])))
        packet.extend(bytearray(struct.pack("f", face.quaternion[2])))
        packet.extend(bytearray(struct.pack("f", face.quaternion[3])))
        packet.extend(bytearray(struct.pack("f", face.euler[0])))
        packet.extend(bytearray(struct.pack("f", face.euler[1])))
        packet.extend(bytearray(struct.pack("f", face.euler[2])))
        packet.extend(bytearray(struct.pack("f", face.translation[0])))
        packet.extend(bytearray(struct.pack("f", face.translation[1])))
        packet.extend(bytearray(struct.pack("f", face.translation[2])))
        for (_,_,c) in face.lms:
            packet.extend(bytearray(struct.pack("f", c)))
        for pt_num, (x,y,_) in enumerate(face.lms):
            packet.extend(bytearray(struct.pack("f", y)))
            packet.extend(bytearray(struct.pack("f", x)))
        for (x,y,z) in face.pts_3d:
            packet.extend(bytearray(struct.pack("f", x)))
            packet.extend(bytearray(struct.pack("f", -y)))
            packet.extend(bytearray(struct.pack("f", -z)))
        if face.current_features is None:
            face.current_features = {}
        for feature in self.features:
            if not feature in face.current_features:
                face.current_features[feature] = 0
            packet.extend(bytearray(struct.pack("f", face.current_features[feature])))
        return packet

    def sendPacket(self, packet):
        try:
            self.sock.sendto(packet, (self.target_ip, self.target_port))
        except:
            self.messageQueue.put("Failed to send packet")
        return
