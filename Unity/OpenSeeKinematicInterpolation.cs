using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace OpenSee {

public class OpenSeeKinematicInterpolation : MonoBehaviour
{
    public float interpolateClamp = 1f;
    
    private Vector3 lastPos;
    private Vector3 lastPosV;
    private double lastT;
    
    private Vector3 currentPos;
    private Vector3 currentPosV;
    private double currentT = 0f;
    
    private Vector3 lastRot = Vector3.zero;
    private Vector3 lastRotV;
    
    private Vector3 currentRot;
    private Vector3 currentRotV;
    
    private float dT;
    private Vector3 coeffPosA;
    private Vector3 coeffPosB;
    private Vector3 coeffRotA;
    private Vector3 coeffRotB;
    
    private double updateT;
    private double interpolateT;
    private int ready = 0;
    
    public void UpdateKI(double newT, Vector3 newPos, Quaternion newRot) {
        Interpolate();
        if (newT - currentT > 3f)
            ready = 0;

        currentPos = transform.localPosition;
        currentRot = OpenSeeIKTarget.PrettyEuler(currentRot, transform.localRotation.eulerAngles);
        float dTNow = (float)(interpolateT - updateT);
        if (interpolateT - updateT > 0 || ready < 1) {
            currentPosV = (currentPos - lastPos) / dT;
            currentRotV = (currentRot - lastRot) / dT;
        } else {
            currentPosV = Vector3.zero;
            currentRotV = Vector3.zero;
        }

        lastPos = currentPos;
        lastPosV = currentPosV;
        lastRot = currentRot;
        lastRotV = currentRotV;
        lastT = currentT;
        
        currentPos = newPos;
        currentRot = OpenSeeIKTarget.PrettyEuler(lastRot, newRot.eulerAngles);
        currentT = newT;
        dT = (float)(currentT - lastT);
        currentPosV = (currentPos - lastPos) / dT;
        currentRotV = (currentRot - lastRot) / dT;

        float dTPow2 = Mathf.Pow(dT, 2);
        float dTPow3 = Mathf.Pow(dT, 3);
        coeffPosA = (- (2f*(currentPosV * dT + 2f * dT * lastPosV - 3f * currentPos + 3f * lastPos)) / dTPow2) / 2f;
        coeffPosB = ((6f*(currentPosV * dT + dT * lastPosV - 2f * currentPos + 2f * lastPos)) / dTPow3)/6f;
        coeffRotA = (- (2f*(currentRotV * dT + 2f * dT * lastRotV - 3f * currentRot + 3f * lastRot)) / dTPow2) / 2f;
        coeffRotB = ((6f*(currentRotV * dT + dT * lastRotV - 2f * currentRot + 2f * lastRot)) / dTPow3)/6f;
        
        updateT = Time.time;
        if (ready <= 2)
            ready++;
    }
    
    float InterpolateValue(float dTNow, float dTNowPow2, float dTNowPow3, float a, float b, float x1, float x2, float v1, float v2) {
        return x1 + v1 * dTNow + dTNowPow2 * a + dTNowPow3 * b;
    }
    
    void Interpolate() {
        interpolateT = Time.time;
        if (interpolateT == updateT || ready <= 2)
            return;
        float dTNow = (float)(interpolateT - updateT);
        if (dTNow >= dT) {
            transform.localPosition = Vector3.LerpUnclamped(lastPos, currentPos, Mathf.Min((float)(dTNow / dT), interpolateClamp));
            transform.localEulerAngles = Vector3.LerpUnclamped(lastRot, currentRot, Mathf.Min((float)(dTNow / dT), interpolateClamp));
        } else {
            float dTNowPow2 = Mathf.Pow(dTNow, 2);
            float dTNowPow3 = Mathf.Pow(dTNow, 3);
            float x = InterpolateValue(dTNow, dTNowPow2, dTNowPow3, coeffPosA.x, coeffPosB.x, lastPos.x, currentPos.x, lastPosV.x, currentPosV.x);
            float y = InterpolateValue(dTNow, dTNowPow2, dTNowPow3, coeffPosA.y, coeffPosB.y, lastPos.y, currentPos.y, lastPosV.y, currentPosV.y);
            float z = InterpolateValue(dTNow, dTNowPow2, dTNowPow3, coeffPosA.z, coeffPosB.z, lastPos.z, currentPos.z, lastPosV.z, currentPosV.z);
            transform.localPosition = new Vector3(x, y, z);
            x = InterpolateValue(dTNow, dTNowPow2, dTNowPow3, coeffRotA.x, coeffRotB.x, lastRot.x, currentRot.x, lastRotV.x, currentRotV.x);
            y = InterpolateValue(dTNow, dTNowPow2, dTNowPow3, coeffRotA.y, coeffRotB.y, lastRot.y, currentRot.y, lastRotV.y, currentRotV.y);
            z = InterpolateValue(dTNow, dTNowPow2, dTNowPow3, coeffRotA.z, coeffRotB.z, lastRot.z, currentRot.z, lastRotV.z, currentRotV.z);
            transform.localEulerAngles = new Vector3(x, y, z);
        }
    }
    
    public void Update() {
        Interpolate();
    }
}

}