using UnityEngine;

namespace OpenSee {

public class OpenSeeIKTarget : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("This is the source of face tracking data.")]
    public OpenSee openSee;
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
    [Tooltip("When enabled, tracking will lag by one video capture frame, but the captured motion information is interpolated between frames, making things look smoother and less jittery and jumpy.")]
    public bool interpolate = true;
    [Tooltip("When set to true, the transform will be updated in FixedUpdate, otherwise it will be updated in Update.")]
    public bool fixedUpdate = false;
    [Header("Information")]
    [Tooltip("This is the current rotation calibration value as euler angles.")]
    public Vector3 rotationOffset = new Vector3(0f, 0f, -90f);
    [Tooltip("This is the current translation calibration.")]
    public Vector3 translationOffset = new Vector3(0f, 0f, 0f);
    [Tooltip("This is the average number of interpolated frames per received tracking data.")]
    public float averageInterpolations = 0f;

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

    void Interpolate() {
        if (!interpolate || interpolateState < 2)
            return;
        float t = Mathf.Clamp((float)interpolationCount / avgInterps, 0f, 0.985f);
        transform.localPosition = Vector3.Lerp(lastT, currentT, t);
        transform.localRotation = Quaternion.Lerp(lastR, currentR, t);
        interpolationCount++;
    }

    Quaternion MirrorQuaternion(Quaternion q) {
        return new Quaternion(-q.x, q.y, q.z, -q.w);
    }

    Vector3 MirrorTranslation(Vector3 v) {
        return new Vector3(-v.x, v.y, v.z);
    }

    void RunUpdate() {
        var openSeeData = openSee.GetOpenSeeData(faceId);
        if (openSeeData == null)
            return;
        if (openSeeData.time > updated) {
            updated = openSeeData.time;
        } else {
            Interpolate();
            return;
        }

        Quaternion convertedQuaternion = new Quaternion(-openSeeData.rawQuaternion.y, -openSeeData.rawQuaternion.x, openSeeData.rawQuaternion.z, openSeeData.rawQuaternion.w);
        Vector3 t = openSeeData.translation;
        t.x = -t.x;
        t.z = -t.z;

        if (calibrate) {
            dR = convertedQuaternion;
            dR = Quaternion.Inverse(dR);
            dT = t;
            rotationOffset = new Vector3(dR.eulerAngles.x, dR.eulerAngles.y, dR.eulerAngles.z);
            translationOffset = new Vector3(dT.x, dT.y, dT.z);
        }

        if (mirrorMotion != lastMirror || (mirrorMotion && calibrate)) {
            dR = Quaternion.Inverse(MirrorQuaternion(Quaternion.Inverse(dR)));
            dT = MirrorTranslation(dT);
            lastMirror = mirrorMotion;
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

        lastT = currentT;
        lastR = currentR;
        if (interpolateState > 0) {
            currentT = Vector3.Lerp(currentT, (t - dT) * translationScale, 1f - smooth);
            currentR = Quaternion.Lerp(transform.localRotation, convertedQuaternion * dR, 1f - smooth);
        } else {
            currentT = (t - dT) * translationScale;
            currentR = convertedQuaternion * dR;
        }
        if (interpolateState < 2)
            interpolateState++;

        if (interpolate) {
            Interpolate();
        } else {
            transform.localPosition = Vector3.Lerp(transform.localPosition, (t - dT) * translationScale, 1f - smooth);
            transform.localRotation = Quaternion.Lerp(transform.localRotation, convertedQuaternion * dR, 1f - smooth);
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
}

}