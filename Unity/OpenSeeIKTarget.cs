using UnityEngine;

public class OpenSeeIKTarget : MonoBehaviour
{
    public OpenSee.OpenSee openSee;
    public bool calibrate = true;
    public int faceId = 0;
    public float translationScale = 0.3f;
    public float smooth = 0.2f;
    double updated = 0.0f;
    Quaternion dR;
    Vector3 dT;

    void Update()
    {
        var openSeeData = openSee.trackingData;
        if (openSeeData == null || openSeeData.Length < 1)
            return;
        int idx = -1;
        int i = 0;
        foreach (var item in openSeeData) {
            if (item.id == faceId)
                idx = i;
            i++;
        }
        if (idx < 0)
            return;
        if (openSeeData[idx].time > updated) {
            updated = openSeeData[idx].time;
        } else {
            return;
        }
        
        // new Quaternion(-openSeeData[idx].rawQuaternion.x, -openSeeData[idx].rawQuaternion.y, openSeeData[idx].rawQuaternion.z, openSeeData[idx].rawQuaternion.w); // Good except for the head tilt in the start being flipped
        Quaternion convertedQuaternion = new Quaternion(-openSeeData[idx].rawQuaternion.y, -openSeeData[idx].rawQuaternion.x, openSeeData[idx].rawQuaternion.z, openSeeData[idx].rawQuaternion.w);

        if (calibrate) {
            calibrate = false;
            dR = convertedQuaternion;//Quaternion.Euler(-openSeeData[idx].rotation.x, openSeeData[idx].rotation.y, -openSeeData[0].rotation.z);
            dR = Quaternion.Inverse(dR);
            dT = openSeeData[idx].translation;
        }
        
        Vector3 v = openSeeData[idx].translation - dT;
        v.z = -v.z;
        transform.localPosition = Vector3.Lerp(transform.localPosition, v * translationScale, 1f - smooth);
        transform.localRotation = Quaternion.Lerp(transform.localRotation, /*Quaternion.Euler(-openSeeData[idx].rotation.x, openSeeData[idx].rotation.y, -openSeeData[idx].rotation.z)*/ convertedQuaternion * dR, 1f - smooth);
    }
}
