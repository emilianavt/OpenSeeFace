using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace OpenSee {

public class OpenSee : MonoBehaviour {
    [Header("UDP server settings")]
    [Tooltip("To listen for remote connections, change this to 0.0.0.0 or your actual IP on the desired interface")]
    public string listenAddress = "127.0.0.1";
    [Tooltip("This is the port the server will listen for tracking packets on")]
    public int listenPort = 11573;

    private const int nPoints = 66;
    private const int packetFrameSize = 8 + 2 * 4 + 1 + 4 + 3 * 4 + 3 * 4 + 4 * 4 + 4 * 68 + 4 * 2 * 68 + 4 * 3 * 70;

    [Header("Tracking data")]
    [Tooltip("This is an informational property that tells you how many packets have been received")]
    public int receivedPackets = 0;
    [Tooltip("This contains the actual tracking data")]
    public OpenSeeData[] trackingData = null;

    [System.Serializable]
    public class OpenSeeData {
        [Tooltip("The time this tracking data was captured at.")]
        public double time;
        [Tooltip("This field tells you how likely it is that the right eye is open.")]
        public float rightEyeOpen;
        [Tooltip("This field tells you how likely it is that the left eye is open.")]
        public float leftEyeOpen;
        [Tooltip("This field tells you if 3D points have been successfully estimated from the 2D points. If this is false, do not rely on pose or 3D data.")]
        public bool got3DPoints;
        [Tooltip("This field contains the error for fitting the original 3D points. It shouldn't matter much, but it it is very high, something is probably wrong")]
        public float fit3DError;
        [Tooltip("This is the rotation vector for the 3D points to turn into the estimated face pose.")]
        public Vector3 rotation;
        [Tooltip("This is the translation vector for the 3D points to turn into the estimated face pose.")]
        public Vector3 translation;
        [Tooltip("This is the raw rotation quaternion calculated from the OpenCV rotation matrix. It does not match Unity's coordinate system, but it still might be useful.")]
        public Quaternion rawQuaternion;
        [Tooltip("This is the raw rotation euler angles calculated by OpenCV from the rotation matrix. It does not match Unity's coordinate system, but it still might be useful.")]
        public Vector3 rawEuler;
        [Tooltip("This field tells you how certain the tracker is.")]
        public float[] confidence;
        [Tooltip("These are the detected face landmarks in image coordinates. There are 68 points. The last too points are pupil points from the gaze tracker.")]
        public Vector2[] points;
        [Tooltip("These are 3D points estimated from the 2D points. The should be rotation and translation compensated. There are 70 points with guesses for the eyeball center positions being added at the end of the 68 2D points.")]
        public Vector3[] points3D;

        public OpenSeeData() {
            confidence = new float[nPoints];
            points = new Vector2[nPoints];
            points3D = new Vector3[nPoints];
        }

        private float readFloat(byte[] b, ref int o) {
            float v = System.BitConverter.ToSingle(b, o);
            o += 4;
            return v;
        }

        private Quaternion readQuaternion(byte[] b, ref int o) {
            float x = readFloat(b, ref o);
            float y = readFloat(b, ref o);
            float z = readFloat(b, ref o);
            float w = readFloat(b, ref o);
            Quaternion q = new Quaternion(x, y, z, w);
            return q;
        }

        private Vector3 readVector3(byte[] b, ref int o) {
            Vector3 v = new Vector3(readFloat(b, ref o), -readFloat(b, ref o), readFloat(b, ref o));
            return v;
        }

        private Vector2 readVector2(byte[] b, ref int o) {
            Vector2 v = new Vector2(readFloat(b, ref o), readFloat(b, ref o));
            return v;
        }

        public void readFromPacket(byte[] b, int o) {
            time = System.BitConverter.ToDouble(b, o);
            o += 8;
            
            rightEyeOpen = readFloat(b, ref o);
            leftEyeOpen = readFloat(b, ref o);

            byte got3D = b[o];
            o++;
            got3DPoints = false;
            if (got3D != 0)
                got3DPoints = true;

            fit3DError = readFloat(b, ref o);
            rawQuaternion = readQuaternion(b, ref o);
            Quaternion convertedQuaternion = new Quaternion(-rawQuaternion.x, rawQuaternion.y, -rawQuaternion.z, rawQuaternion.w);
            rawEuler = readVector3(b, ref o);

            rotation = rawEuler;
            rotation.z = (rotation.z + 90) % 360;
            rotation.x = -(rotation.x + 180) % 360;

            float x = readFloat(b, ref o);
            float y = readFloat(b, ref o);
            float z = readFloat(b, ref o);
            translation = new Vector3(y, -x, z);

            for (int i = 0; i < nPoints; i++) {
                confidence[i] = readFloat(b, ref o);
            }
            
            // Gaze tracker results are still bad, ignore them
            readFloat(b, ref o); readFloat(b, ref o);

            for (int i = 0; i < nPoints; i++) {
                points[i] = readVector2(b, ref o);
            }
            
            // Gaze tracker results are still bad, ignore them
            readVector2(b, ref o); readVector2(b, ref o);

            for (int i = 0; i < nPoints; i++) {
                points3D[i] = readVector3(b, ref o);
            }
            
            // Gaze tracker results are still bad, ignore them
            readVector3(b, ref o); readVector3(b, ref o); readVector3(b, ref o); readVector3(b, ref o);
        }
    }

    private Socket socket;
    private byte[] buffer;
    private Thread receiveThread = null;
    private volatile bool stopReception = false;

    void performReception() {
        EndPoint senderRemote = new IPEndPoint(IPAddress.Any, 0);
        while (!stopReception) {
            try {
                int receivedBytes = socket.ReceiveFrom(buffer, SocketFlags.None, ref senderRemote);
                if (receivedBytes < 1 || receivedBytes % packetFrameSize != 0) {
                    continue;
                }
                receivedPackets++;
                int i = 0;
                OpenSeeData[] newData = new OpenSeeData[receivedBytes / packetFrameSize];
                for (int offset = 0; offset < receivedBytes; offset += packetFrameSize) {
                    newData[i] = new OpenSeeData();
                    newData[i].readFromPacket(buffer, offset);
                    i++;
                }
                trackingData = newData;
            } catch { }
        }
    }

    void Start () {
        buffer = new byte[65535];

        socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        IPAddress ip;
        IPAddress.TryParse(listenAddress, out ip);
        socket.Bind(new IPEndPoint(ip, listenPort));
        socket.ReceiveTimeout = 15;

        receiveThread = new Thread(() => performReception());
        receiveThread.Start();
    }

    void Update () {
	}

    void OnApplicationQuit() {
        if (receiveThread != null) {
            stopReception = true;
            receiveThread.Join();
            stopReception = false;
        }
    }
}

}