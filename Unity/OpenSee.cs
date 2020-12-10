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

    private const int nPoints = 68;
    private const int packetFrameSize = 8 + 4 + 2 * 4 + 2 * 4 + 1 + 4 + 3 * 4 + 3 * 4 + 4 * 4 + 4 * 68 + 4 * 2 * 68 + 4 * 3 * 70 + 4 * 14;

    [Header("Tracking data")]
    [Tooltip("This is an informational property that tells you how many packets have been received")]
    public int receivedPackets = 0;
    [Tooltip("This contains the actual tracking data")]
    public OpenSeeData[] trackingData = null;
    
    public bool listening { get; private set; } = false;
    
    [HideInInspector]
    public float maxFit3DError = 100f;

    [System.Serializable]
    public class OpenSeeData {
        [Tooltip("The time this tracking data was captured at.")]
        public double time;
        [Tooltip("This is the id of the tracked face. When tracking multiple faces, they might get reordered due to faces coming and going, but as long as tracking is not lost on a face, its id should stay the same. Face ids depend only on the order of first detection and locations of the faces.")]
        public int id;
        [Tooltip("This field gives the resolution of the camera or video being tracked.")]
        public Vector2 cameraResolution;
        [Tooltip("This field tells you how likely it is that the right eye is open.")]
        public float rightEyeOpen;
        [Tooltip("This field tells you how likely it is that the left eye is open.")]
        public float leftEyeOpen;
        [Tooltip("This field contains the rotation of the right eyeball.")]
        public Quaternion rightGaze;
        [Tooltip("This field contains the rotation of the left eyeball.")]
        public Quaternion leftGaze;
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
        [Tooltip("This field contains a number of action unit like features.")]
        public OpenSeeFeatures features;
        
        [System.Serializable]
        public class OpenSeeFeatures {
            [Tooltip("This field indicates whether the left eye is opened(0) or closed (-1). A value of 1 means open wider than normal.")]
            public float EyeLeft;
            [Tooltip("This field indicates whether the right eye is opened(0) or closed (-1). A value of 1 means open wider than normal.")]
            public float EyeRight;
            [Tooltip("This field indicates how steep the left eyebrow is, compared to the median steepness.")]
            public float EyebrowSteepnessLeft;
            [Tooltip("This field indicates how far up or down the left eyebrow is, compared to its median position.")]
            public float EyebrowUpDownLeft;
            [Tooltip("This field indicates how quirked the left eyebrow is, compared to its median quirk.")]
            public float EyebrowQuirkLeft;
            [Tooltip("This field indicates how steep the right eyebrow is, compared to the average steepness.")]
            public float EyebrowSteepnessRight;
            [Tooltip("This field indicates how far up or down the right eyebrow is, compared to its median position.")]
            public float EyebrowUpDownRight;
            [Tooltip("This field indicates how quirked the right eyebrow is, compared to its median quirk.")]
            public float EyebrowQuirkRight;
            [Tooltip("This field indicates how far up or down the left mouth corner is, compared to its median position.")]
            public float MouthCornerUpDownLeft;
            [Tooltip("This field indicates how far in or out the left mouth corner is, compared to its median position.")]
            public float MouthCornerInOutLeft;
            [Tooltip("This field indicates how far up or down the right mouth corner is, compared to its median position.")]
            public float MouthCornerUpDownRight;
            [Tooltip("This field indicates how far in or out the right mouth corner is, compared to its median position.")]
            public float MouthCornerInOutRight;
            [Tooltip("This field indicates how open or closed the mouth is, compared to its median pose.")]
            public float MouthOpen;
            [Tooltip("This field indicates how wide the mouth is, compared to its median pose.")]
            public float MouthWide;
        }

        public OpenSeeData() {
            confidence = new float[nPoints];
            points = new Vector2[nPoints];
            points3D = new Vector3[nPoints + 2];
        }
        
        private Vector3 swapX(Vector3 v) {
            v.x = -v.x;
            return v;
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
            id = System.BitConverter.ToInt32(b, o);
            o += 4;

            cameraResolution = readVector2(b, ref o);
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
            rotation.z = (rotation.z - 90) % 360;
            rotation.x = -(rotation.x + 180) % 360;

            float x = readFloat(b, ref o);
            float y = readFloat(b, ref o);
            float z = readFloat(b, ref o);
            translation = new Vector3(-y, x, -z);

            for (int i = 0; i < nPoints; i++) {
                confidence[i] = readFloat(b, ref o);
            }

            for (int i = 0; i < nPoints; i++) {
                points[i] = readVector2(b, ref o);
            }

            for (int i = 0; i < nPoints + 2; i++) {
                points3D[i] = readVector3(b, ref o);
            }
            
            rightGaze = Quaternion.LookRotation(swapX(points3D[66]) - swapX(points3D[68])) * Quaternion.AngleAxis(180, Vector3.right) * Quaternion.AngleAxis(180, Vector3.forward);
            leftGaze = Quaternion.LookRotation(swapX(points3D[67]) - swapX(points3D[69])) * Quaternion.AngleAxis(180, Vector3.right) * Quaternion.AngleAxis(180, Vector3.forward);
            
            features = new OpenSeeFeatures();
            features.EyeLeft = readFloat(b, ref o);
            features.EyeRight = readFloat(b, ref o);
            features.EyebrowSteepnessLeft = readFloat(b, ref o);
            features.EyebrowUpDownLeft = readFloat(b, ref o);
            features.EyebrowQuirkLeft = readFloat(b, ref o);
            features.EyebrowSteepnessRight = readFloat(b, ref o);
            features.EyebrowUpDownRight = readFloat(b, ref o);
            features.EyebrowQuirkRight = readFloat(b, ref o);
            features.MouthCornerUpDownLeft = readFloat(b, ref o);
            features.MouthCornerInOutLeft = readFloat(b, ref o);
            features.MouthCornerUpDownRight = readFloat(b, ref o);
            features.MouthCornerInOutRight = readFloat(b, ref o);
            features.MouthOpen = readFloat(b, ref o);
            features.MouthWide = readFloat(b, ref o);
        }
    }

    private Dictionary<int, OpenSeeData> openSeeDataMap;
    private Socket socket;
    private byte[] buffer;
    private Thread receiveThread = null;
    private volatile bool stopReception = false;

    public OpenSeeData GetOpenSeeData(int faceId) {
        if (openSeeDataMap == null)
            return null;
        if (!openSeeDataMap.ContainsKey(faceId))
            return null;
        return openSeeDataMap[faceId];
    }

    void performReception() {
        EndPoint senderRemote = new IPEndPoint(IPAddress.Any, 0);
        listening = true;
        while (!stopReception) {
            try {
                int receivedBytes = socket.ReceiveFrom(buffer, SocketFlags.None, ref senderRemote);
                if (receivedBytes < 1 || receivedBytes % packetFrameSize != 0) {
                    continue;
                }
                receivedPackets++;
                int i = 0;
                for (int offset = 0; offset < receivedBytes; offset += packetFrameSize) {
                    OpenSeeData newData = new OpenSeeData();
                    newData.readFromPacket(buffer, offset);
                    openSeeDataMap[newData.id] = newData;
                    i++;
                }
                trackingData = new OpenSeeData[openSeeDataMap.Count];
                openSeeDataMap.Values.CopyTo(trackingData, 0);
            } catch { }
        }
    }

    void Start () {
        if (openSeeDataMap == null)
            openSeeDataMap = new Dictionary<int, OpenSeeData>();
        buffer = new byte[65535];

        if (socket == null) {
            socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            IPAddress ip;
            IPAddress.TryParse(listenAddress, out ip);
            socket.Bind(new IPEndPoint(ip, listenPort));
            socket.ReceiveTimeout = 15;
        }

        receiveThread = new Thread(() => performReception());
        receiveThread.Start();
    }

    void Update () {
        if (receiveThread != null && !receiveThread.IsAlive) {
            Start();
        }
	}
    
    void EndReceiver() {
        if (receiveThread != null) {
            stopReception = true;
            receiveThread.Join();
            stopReception = false;
        }
        if (socket != null)
            socket.Close();
    }

    void OnApplicationQuit() {
        EndReceiver();
    }
    
    void OnDestroy() {
        EndReceiver();
    }
}

}