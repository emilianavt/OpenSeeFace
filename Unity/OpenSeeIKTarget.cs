using UnityEngine;

namespace OpenSee {

[DefaultExecutionOrder(-1)]
public class OpenSeeIKTarget : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("This is the source of face tracking data.")]
    public OpenSee openSee;
    [Tooltip("When this is set to another game object, its kinematic interpolation component will get updated from values of this.")]
    public OpenSeeKinematicInterpolation kinematicInterpolation;
    [Tooltip("This can be used to skip the first few tracking frames.")]
    public int skipFirst = 5;
    [Tooltip("When set to true, the next rotation and translation values will be set as zero rotation and zero translation.")]
    public bool calibrate = true;
    [Tooltip("This sets which face id to look for in the OpenSee data.")]
    public int faceId = 0;
    [Tooltip("When this is enabled, rotations and movement are mirrored.")]
    public bool mirrorMotion = false;
    [Tooltip("Often, the translation vector's scale is too high. Setting this somewhere between 0.05 to 0.3 seems stabilize things, but it also reduces the range of motion.")]
    public float translationScale = 0.3f;
    [Tooltip("This smoothed out the detected motion. Values can be between 0 (no smoothing) and 1 (no motion). The appropriate value probably depends on the camera's frame rate.")]
    [Range(0, 1)]
    public float smooth = 0.2f;
    [Tooltip("When set to true, a Kalman filter is used for smoothing.")]
    public bool kalman = false;
    [Tooltip("When enabled, tracking will lag by one video capture frame, but the captured motion information is interpolated between frames, making things look smoother and less jittery and jumpy.")]
    public bool interpolate = true;
    [Tooltip("When set to true, the transform will be updated in FixedUpdate, otherwise it will be updated in Update.")]
    public bool fixedUpdate = false;
    [Tooltip("When set to true, the IK target's position and location will slowly drift back to its neutral pose.")]
    public bool driftBack = false;
    [Tooltip("This sets the strength of the drifting effect. At 0 there is now drifting, while the IK target will be static at 1.")]
    public float driftFactor = 0.005f;
    [Tooltip("This sets the minimum angle difference between two poses after which the pose will be considered an outlier.")]
    [Range(0, 360)]
    public float outlierThresholdAngle = 35f;
    [Tooltip("This sets the number of seconds for which angle outliers are ignored before being accepted. Setting this to zero will disable outlier skipping.")]
    public float outlierSkipPeriod = 0.4f;
    [Tooltip("This sets the minimum distance difference between two poses after which the pose will be considered an outlier.")]
    public float outlierThresholdDistance = 2f;
    [Tooltip("This sets the number of seconds for which distance outliers are ignored before being accepted. Once this period elapses, automatic recalibration is triggered. Setting this to zero will disable outlier skipping.")]
    public float outlierSkipPeriodDistance = 0.4f;
    [Tooltip("If set to true, persistent positional outliers will cause automatic recalibration.")]
    public bool outlierRecalibrate = false;
    [Tooltip("If set to true, recalibration caused by positional outliers will keep the current position and rotation of the head.")]
    public bool outlierKeepOffset = false;
    [Header("Information")]
    [Tooltip("This is the number of received tracking frames.")]
    public int trackingFrames = 0;
    [Tooltip("This is the current rotation calibration value as euler angles.")]
    public Vector3 rotationOffset = new Vector3(0f, 0f, -90f);
    [Tooltip("This is the current translation calibration.")]
    public Vector3 translationOffset = new Vector3(0f, 0f, 0f);
    [Tooltip("This is the average number of interpolated frames per received tracking data.")]
    public float averageInterpolations = 0f;
    [Tooltip("This is the number of outlier skipped tracking data frames so far.")]
    public int outlierSkips = 0;
    [Tooltip("This is the number of distance outlier skipped tracking data frames so far.")]
    public int outlierSkipsDistance = 0;
    [Tooltip("This is the number of times a recalibration was triggered due to positional outliers.")]
    public int outlierCalibrations = 0;

    private double updated = 0.0f;
    private Quaternion dR = Quaternion.Euler(0f, 0f, -90f);
    private Vector3 dT = new Vector3(0f, 0f, 0f);

    private int interpolationCount = 1;
    private int interpolateState = 0;
    private float avgInterps = 1f;
    private Quaternion lastR;
    private Vector3 lastT;
    private Quaternion currentR;
    private Vector3 currentT;
    private bool lastMirror = false;

    private bool gotAccepted = false;
    private Quaternion lastAccepted;
    private bool skipped = false;
    private float skipStartTime = -1f;
    
    private bool gotAcceptedPosition = false;
    private Vector3 lastAcceptedPosition;
    private bool skippedPosition = false;
    private float skipStartTimePosition = -1f;
    
    // Kalman
    private float frameTime = 0f;
    private bool kalmanInit = false;
    private float kalmanSmooth = -1f;
    private KF kalmanPVX;
    private KF kalmanPVY;
    private KF kalmanPVZ;
    private KF kalmanRVX;
    private KF kalmanRVY;
    private KF kalmanRVZ;
    private Vector3 kalmanLastRot = Vector3.zero;
    
    public void Calibrate() {
        calibrate = true;
    }
    
    public static Vector3 PrettyEuler(Vector3 last, Vector3 current) {
        // Ew, but how would you do componentwise filtering/interpolation with quaternions?
        if (Mathf.Abs(current.x - last.x) > Mathf.Abs((current.x - 360f) - last.x))
            current.x -= 360f;
        else if (Mathf.Abs(current.x - last.x) > Mathf.Abs((current.x + 360f) - last.x))
            current.x += 360f;
        if (Mathf.Abs(current.y - last.y) > Mathf.Abs((current.y - 360f) - last.y))
            current.y -= 360f;
        else if (Mathf.Abs(current.y - last.y) > Mathf.Abs((current.y + 360f) - last.y))
            current.x += 360f;
        if (Mathf.Abs(current.z - last.z) > Mathf.Abs((current.z - 360f) - last.z))
            current.z -= 360f;
        else if (Mathf.Abs(current.z - last.z) > Mathf.Abs((current.z + 360f) - last.z))
            current.z += 360f;
        return current;
    }
    
    Vector3 FilterPos(Vector3 a, Vector3 b) {
        if (!kalman || frameTime > 1f)
            return Vector3.Lerp(a, b, 1f - smooth);
        else {
            Vector3 v = (b - a) / frameTime;
            v.x = kalmanPVX.Update(v.x);
            v.y = kalmanPVY.Update(v.y);
            v.z = kalmanPVZ.Update(v.z);
            return a + v * frameTime;
        }
    }
    
    Quaternion FilterRot(Quaternion a, Quaternion b) {
        if (!kalman || frameTime > 1f)
            return Quaternion.Lerp(a, b, 1f - smooth);
        else {
            Vector3 rotA = PrettyEuler(kalmanLastRot, a.eulerAngles);
            Vector3 rotB = PrettyEuler(rotA, b.eulerAngles);
            kalmanLastRot = rotA;
            Vector3 v = Mathf.Deg2Rad * (rotB - rotA) / frameTime;
            v.x = kalmanRVX.Update(v.x);
            v.y = kalmanRVY.Update(v.y);
            v.z = kalmanRVZ.Update(v.z);
            return Quaternion.Euler(rotA + Mathf.Rad2Deg * v * frameTime);
        }
    }

    void Interpolate() {
        if (!interpolate || interpolateState < 2)
            return;
        float t = Mathf.Clamp((float)interpolationCount / avgInterps, 0f, 1.5f);
        transform.localPosition = Vector3.SlerpUnclamped(lastT, currentT, t);
        transform.localRotation = Quaternion.SlerpUnclamped(lastR, currentR, t);
        interpolationCount++;
        //transform.localPosition = new Vector3(0.3f * ((Time.time/3f) % 3f) - 0.5f, 0.3f * Mathf.Sin(Mathf.Rad2Deg * ((Time.time/5f) % 3f)), 0f);
        //transform.localRotation = Quaternion.identity * Quaternion.AngleAxis(20f * Mathf.Sin(Mathf.Rad2Deg * ((Time.time/5f) % 3f)) - 10f, Vector3.up);
    }

    Quaternion MirrorQuaternion(Quaternion q) {
        return new Quaternion(-q.x, q.y, q.z, -q.w);
    }

    Vector3 MirrorTranslation(Vector3 v) {
        return new Vector3(-v.x, v.y, v.z);
    }

    void RunUpdate() {
        var openSeeData = openSee.GetOpenSeeData(faceId);
        if (openSeeData == null || openSeeData.fit3DError > openSee.maxFit3DError)
            return;
        if (openSeeData.time > updated) {
            frameTime = (float)(openSeeData.time - updated);
            updated = openSeeData.time;
            trackingFrames++;
            if (trackingFrames < skipFirst)
                return;
        } else {
            Interpolate();
            return;
        }
        
        if (!kalmanInit) {
            kalmanPVX = new KF(smooth);
            kalmanPVY = new KF(smooth);
            kalmanPVZ = new KF(smooth);
            kalmanRVX = new KF(smooth);
            kalmanRVY = new KF(smooth);
            kalmanRVZ = new KF(smooth);
            kalmanInit = true;
        }
        if (smooth != kalmanSmooth) {
            kalmanPVX.SetSmoothness(smooth);
            kalmanPVY.SetSmoothness(smooth);
            kalmanPVZ.SetSmoothness(smooth);
            kalmanRVX.SetSmoothness(smooth);
            kalmanRVY.SetSmoothness(smooth);
            kalmanRVZ.SetSmoothness(smooth);
            kalmanSmooth = smooth;
        }

        Quaternion convertedQuaternion = new Quaternion(-openSeeData.rawQuaternion.y, -openSeeData.rawQuaternion.x, openSeeData.rawQuaternion.z, openSeeData.rawQuaternion.w);
        Vector3 t = openSeeData.translation;
        t.x = -t.x;
        t.z = -t.z;
        Vector3 convertedTranslation = new Vector3(t.x, t.y, t.z);
        
        // Check for angular outliers
        if (gotAccepted) {
            float angularDifference = Quaternion.Angle(lastAccepted, convertedQuaternion);
            if (angularDifference >= outlierThresholdAngle) {
                if (!skipped) {
                    if (outlierSkipPeriod >= 0.0000001f) {
                        skipStartTime = Time.time;
                        skipped = true;
                    }
                } else {
                    if (Time.time > skipStartTime + outlierSkipPeriod) {
                        lastAccepted = convertedQuaternion;
                        skipped = false;
                    }
                }
            } else {
                skipped = false;
                lastAccepted = convertedQuaternion;
            }
        } else {
            lastAccepted = convertedQuaternion;
            skipped = false;
        }
        gotAccepted = true;
        if (skipped) {
            outlierSkips += 1;
            return;
        }
        
        // Check for positional outliers
        bool keepOffset = false;
        if (gotAcceptedPosition) {
            float distance = Vector3.Distance(t, lastAcceptedPosition);
            if (distance > outlierThresholdDistance) {
                if (!skippedPosition) {
                    if (outlierSkipPeriodDistance >= 0.0000001f) {
                        skipStartTimePosition = Time.time;
                        skippedPosition = true;
                    }
                } else {
                    if (Time.time > skipStartTimePosition + outlierSkipPeriodDistance) {
                        lastAcceptedPosition = t;
                        skippedPosition = false;
                        if (outlierRecalibrate) {
                            calibrate = true;
                            keepOffset = outlierKeepOffset;
                            outlierCalibrations += 1;
                        }
                    }
                }
            } else {
                skippedPosition = false;
                lastAcceptedPosition = t;
            }
        } else {
            skippedPosition = false;
            lastAcceptedPosition = t;
        }
        gotAcceptedPosition = true;
        if (skippedPosition) {
            outlierSkipsDistance += 1;
            return;
        }

        if (calibrate) {
            dR = convertedQuaternion;
            dR = Quaternion.Inverse(dR);
            dT = t;
            rotationOffset = new Vector3(dR.eulerAngles.x, dR.eulerAngles.y, dR.eulerAngles.z);
            translationOffset = new Vector3(dT.x, dT.y, dT.z);
            lastAcceptedPosition = translationOffset;
            skippedPosition = false;
        }

        if (mirrorMotion != lastMirror || (mirrorMotion && calibrate)) {
            dR = Quaternion.Inverse(MirrorQuaternion(Quaternion.Inverse(dR)));
            dT = MirrorTranslation(dT);
            lastMirror = mirrorMotion;
        }
        if (calibrate && keepOffset) {
            dR = Quaternion.Inverse(Quaternion.Inverse(transform.localRotation) * Quaternion.Inverse(dR));
            dT -= transform.localPosition;
        }
        calibrate = false;

        if (mirrorMotion) {
            convertedQuaternion = MirrorQuaternion(convertedQuaternion);
            t = MirrorTranslation(t);
        }
        
        if (interpolateState > 1)
            avgInterps = Mathf.Lerp(avgInterps, (float)interpolationCount, 0.15f);
        interpolationCount = 0;
        averageInterpolations = avgInterps;
        
        lastT = transform.localPosition;
        lastR = transform.localRotation;
        if (interpolateState > 0) {
            currentT = FilterPos(currentT, (t - dT) * translationScale);
            currentR = FilterRot(transform.localRotation, convertedQuaternion * dR);
        } else {
            currentT = (t - dT) * translationScale;
            currentR = convertedQuaternion * dR;
        }
        if (interpolateState < 2)
            interpolateState++;

        if (interpolate) {
            Interpolate();
        } else {
            transform.localPosition = FilterPos(transform.localPosition, (t - dT) * translationScale);
            transform.localRotation = FilterRot(transform.localRotation, convertedQuaternion * dR);
        }
        if (kinematicInterpolation != null && kinematicInterpolation.gameObject != gameObject) {
            if (interpolate)
                kinematicInterpolation.UpdateKI(updated, currentT, currentR);
            else
                kinematicInterpolation.UpdateKI(updated, transform.localPosition, transform.localRotation);
        }
        
        if (driftBack) {
            if (mirrorMotion) {
                dT = Vector3.Lerp(dT, MirrorTranslation(convertedTranslation), driftFactor);
                dR = Quaternion.Lerp(dR, Quaternion.Inverse(convertedQuaternion), driftFactor);
            } else {
                dT = Vector3.Lerp(dT, convertedTranslation, driftFactor);
                dR = Quaternion.Lerp(dR, Quaternion.Inverse(convertedQuaternion), driftFactor);
            }
            rotationOffset = new Vector3(dR.eulerAngles.x, dR.eulerAngles.y, dR.eulerAngles.z);
            translationOffset = new Vector3(dT.x, dT.y, dT.z);
       }
    }

    void FixedUpdate()
    {
        if (fixedUpdate)
            RunUpdate();
    }

    void Update()
    {
        if (!fixedUpdate)
            RunUpdate();
    }
    
    class KF {
        private float R, P, Q, x;

        public KF(float smooth) {
            P = 1f;
            x = 0f;
            SetSmoothness(smooth);
        }
        
        public void SetSmoothness(float smooth) {
            P = 1f;
            Q = 0.3f;// * (Mathf.Exp(3f * smooth) + 1f);
            R = 0.000001f + Mathf.Exp(smooth * 5f);
        }

        public float Update(float input) {
            //x = x * A;
            //P = A * P * A;
            
            float K = P / (P + R);
            x = x + K * (input - x);
            //Debug.Log("P = " + P + " 1-K = " + (1f - K));
            P = (1f - K) * P + Q;

            return x;
        }
    }
}

}