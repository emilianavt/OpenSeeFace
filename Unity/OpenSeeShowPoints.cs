using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace OpenSee {

public class OpenSeeShowPoints : MonoBehaviour {
    public OpenSee openSee = null;
    public int faceId = 0;
    public bool show3DPoints = true;
    public bool applyTranslation = false;
    public bool applyRotation = false;
    [Range(0, 1)]
    public float minConfidence = 0.20f;
    private OpenSee.OpenSeeData openSeeData;
    private GameObject[] gameObjects;
    private GameObject centerBall;
    private double updated = 0.0;

	void Start () {
        if (openSee == null) {
            openSee = GetComponent<OpenSee>();
        }
        gameObjects = new GameObject[70];
        for (int i = 0; i < 70; i++) {
            gameObjects[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            gameObjects[i].name = "Point " + (i + 1);
            gameObjects[i].transform.SetParent(transform);
            gameObjects[i].transform.localScale = new Vector3(0.025f, 0.025f, 0.025f);
            if (i >= 68) {
                GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                cylinder.transform.SetParent(gameObjects[i].transform);
                cylinder.transform.localEulerAngles = new Vector3(90f, 0f, 0f);
                cylinder.transform.localPosition = new Vector3(0f, 0f, -4f);
                cylinder.transform.localScale = new Vector3(1f, 4f, 1f);
            }
        }
        centerBall = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        centerBall.name = "Center";
        centerBall.transform.SetParent(transform);
        centerBall.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
	}

	void Update () {
        if (!openSee)
            return;
        /*openSeeData = openSee.trackingData;
        if (openSeeData == null || openSeeData.Length < 1)
            return;*/
        openSeeData = openSee.GetOpenSeeData(faceId);
        if (openSeeData == null)
            return;
        if (openSeeData.time > updated) {
            updated = openSeeData.time;
        } else {
            return;
        }
        if (show3DPoints) {
            centerBall.gameObject.SetActive(false);
            for (int i = 0; i < 70; i++) {
                if (openSeeData.got3DPoints && (i >= 68 || openSeeData.confidence[i] > minConfidence)) {
                    Renderer renderer = gameObjects[i].GetComponent<Renderer>();
                    Vector3 pt = openSeeData.points3D[i];
                    pt.x = -pt.x;
                    gameObjects[i].transform.localPosition = pt;
                    if (i < 68)
                        renderer.material.SetColor("_Color", Color.Lerp(Color.red, Color.green, openSeeData.confidence[i]));
                    else {
                        if (i == 68)
                            gameObjects[i].transform.localRotation = openSeeData.rightGaze;
                        else
                            gameObjects[i].transform.localRotation = openSeeData.leftGaze;
                    }
                } else {
                    Renderer renderer = gameObjects[i].GetComponent<Renderer>();
                    renderer.material.SetColor("_Color", Color.cyan);
                }
            }
            if (applyTranslation) {
                Vector3 v = openSeeData.translation;
                v.x = -v.x;
                v.z = -v.z;
                transform.localPosition = v;
            }
            if (applyRotation) {
                Quaternion offset = Quaternion.Euler(0f, 0f, -90f);
                Quaternion convertedQuaternion = new Quaternion(-openSeeData.rawQuaternion.y, -openSeeData.rawQuaternion.x, openSeeData.rawQuaternion.z, openSeeData.rawQuaternion.w) * offset;
                transform.localRotation = convertedQuaternion;
            }
        } else {
            centerBall.gameObject.SetActive(false);
            Vector3 center = new Vector3(0.0f, 0.0f, 0.0f);
            float minX = 10000.0f;
            float minY = 10000.0f;
            float maxX = -1.0f;
            float maxY = -1.0f;
            for (int i = 0; i < 66; i++) {
                float x = openSeeData.points[i].x;
                float y = -openSeeData.points[i].y;
                if (minX > x) {
                    minX = x;
                }
                if (minY > y) {
                    minY = y;
                }
                if (maxX < x) {
                    maxX = x;
                }
                if (maxY < y) {
                    maxY = y;
                }
                center += new Vector3(x, y, 0.0f);
            }
            center = center / 66;
            center = center - new Vector3(minX, minY, 0.0f);
            center.x = center.x / (maxX - minX);
            center.y = center.y / (maxX - minX);
            center.z = 0.5f;
            centerBall.transform.localPosition = center;

            for (int i = 0; i < 66; i++) {
                Renderer renderer = gameObjects[i].GetComponent<Renderer>();
                renderer.material.SetColor("_Color", Color.Lerp(Color.red, Color.green, openSeeData.confidence[i]));
                float x = openSeeData.points[i].x;
                float y = -openSeeData.points[i].y;
                Vector3 position = new Vector3(x, y, 0.0f);
                position = position - new Vector3(minX, minY, 0.0f);
                position.x = position.x / (maxX - minX);
                position.y = position.y / (maxX - minX);
                gameObjects[i].transform.localPosition = position;
            }
        }
	}
}

}