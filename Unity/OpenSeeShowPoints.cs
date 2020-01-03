using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace OpenSee {

public class OpenSeeShowPoints : MonoBehaviour {
    public OpenSee openSee = null;
    public bool show3DPoints = true;
    public bool applyTranslation = false;
    public bool applyRotation = false;
    public float minConfidence = 0.20f;
    private OpenSee.OpenSeeData[] openSeeData;
    private GameObject[] gameObjects;
    private GameObject centerBall;
    private double updated = 0.0;

	void Start () {
        if (openSee == null) {
            openSee = GetComponent<OpenSee>();
        }
        gameObjects = new GameObject[68];
        for (int i = 0; i < 68; i++) {
            gameObjects[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            gameObjects[i].name = "Point " + (i + 1);
            gameObjects[i].transform.SetParent(transform);
            gameObjects[i].transform.localScale = new Vector3(0.025f, 0.025f, 0.025f);
        }
        centerBall = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        centerBall.name = "Center";
        centerBall.transform.SetParent(transform);
        centerBall.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
	}
    
	void Update () {
        if (!openSee)
            return;
        openSeeData = openSee.trackingData;
        if (openSeeData == null || openSeeData.Length < 1)
            return;
        if (openSeeData[0].time > updated) {
            updated = openSeeData[0].time;
        } else {
            return;
        }
        if (show3DPoints) {
            centerBall.gameObject.SetActive(false);
            for (int i = 0; i < 66; i++) {
                if (openSeeData[0].got3DPoints && openSeeData[0].confidence[i] > minConfidence) {
                    Renderer renderer = gameObjects[i].GetComponent<Renderer>();
                    renderer.material.SetColor("_Color", Color.Lerp(Color.red, Color.green, openSeeData[0].confidence[i]));
                    gameObjects[i].transform.localPosition = openSeeData[0].points3D[i];
                } else {
                    Renderer renderer = gameObjects[i].GetComponent<Renderer>();
                    renderer.material.SetColor("_Color", Color.cyan);
                }
            }
            if (applyTranslation) {
                transform.position = openSeeData[0].translation;
            }
            if (applyRotation) {
                transform.eulerAngles = openSeeData[0].rotation;
            }
        } else {
            centerBall.gameObject.SetActive(false);
            Vector3 center = new Vector3(0.0f, 0.0f, 0.0f);
            float minX = 10000.0f;
            float minY = 10000.0f;
            float maxX = -1.0f;
            float maxY = -1.0f;
            for (int i = 0; i < 66; i++) {
                float x = openSeeData[0].points[i].x;
                float y = -openSeeData[0].points[i].y;
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
                renderer.material.SetColor("_Color", Color.Lerp(Color.red, Color.green, openSeeData[0].confidence[i]));
                float x = openSeeData[0].points[i].x;
                float y = -openSeeData[0].points[i].y;
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