using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace OpenSee {

public class OpenSeeShowPoints : MonoBehaviour {
    public OpenSee openSee = null;
    public int faceId = 0;
    public bool only30Points = false;
    public bool show3DPoints = true;
    public bool applyTranslation = false;
    public bool applyRotation = false;
    [Range(0, 1)]
    public float minConfidence = 0.20f;
    public bool showGaze = true;
    public Material material;
    public bool showLines = false;
    public float lineWidth = 0.01f;
    public Material lineMaterial;
    public bool receiveShadows = false;
    
    private OpenSee.OpenSeeData openSeeData;
    private GameObject[] gameObjects;
    private LineRenderer[] lineRenderers;
    private GameObject centerBall;
    private double updated = 0.0;
    private int total = 70;
    
    private int[] lines = new int[]{/* Contour */ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, /* Eyebrows */ 18, 19, 20, 21, -1, 23, 24, 25, 26, -1, /* Nose */ 28, 29, 30, 33, 32, 33, 34, 35, -1, /* Eye */ 37, 38, 39, 40, 41, 36, /* Eye */ 43, 44, 45, 46, 47, 42, /* Mouth */ 49, 50, 51, 52, 62, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 58, 58, 62};
    
    private HashSet<int> pt30Set = new HashSet<int> { 0, 2, 5, 8, 11, 14, 16, 17, 19, 21, 22, 24, 26, 27, 30, 33, 36, 37, 39, 40, 42, 43, 45, 46, 50, 55, 58, 60, 62, 64 };
    private int[] linesPt30Set = new int[]{/* Contour */ 2, -1, 5, -1, -1, 8, -1, -1, 11, -1, -1, 14, -1, -1, 16, -1, -1, /* Eyebrows */ 19, -1, 21, -1, -1, 24, -1, 26, -1, -1, /* Nose */ 30, -1, -1, 33, -1, -1, -1, -1, -1, /* Eye */ 37, 39, -1, 40, 36, -1, /* Eye */ 43, 45, -1, 46, 42, -1, /* Mouth */ -1, -1, 62, -1, -1, -1, -1, 58, -1, -1, 60, -1, 62, -1, 64, -1, 58, -1, 58, 62};

	void Start () {
        if (openSee == null) {
            openSee = GetComponent<OpenSee>();
        }
        gameObjects = new GameObject[70];
        lineRenderers = new LineRenderer[68];
        if (lineMaterial == null)
            showLines = false;
        if (!showGaze)
            total = 66;
        for (int i = 0; i < total; i++) {
            gameObjects[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            gameObjects[i].name = "Point " + (i + 1);
            if (material != null)
                gameObjects[i].GetComponent<Renderer>().material = material;
            gameObjects[i].GetComponent<Renderer>().receiveShadows = receiveShadows;
            gameObjects[i].layer = gameObject.layer;
            gameObjects[i].transform.SetParent(transform);
            gameObjects[i].transform.localScale = new Vector3(0.025f, 0.025f, 0.025f);
            if (i >= 68) {
                GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                if (material != null)
                    cylinder.GetComponent<Renderer>().material = material;
                cylinder.GetComponent<Renderer>().receiveShadows = receiveShadows;
                cylinder.layer = gameObject.layer;
                cylinder.transform.SetParent(gameObjects[i].transform);
                cylinder.transform.localEulerAngles = new Vector3(90f, 0f, 0f);
                cylinder.transform.localPosition = new Vector3(0f, 0f, -4f);
                cylinder.transform.localScale = new Vector3(1f, 4f, 1f);
            }
            if (only30Points && i < 66 && !pt30Set.Contains(i))
                gameObjects[i].SetActive(false);
        }
        for (int i = 0; i < 68; i++) {
                if (i == 66) {
                    GameObject lineGameObject = new GameObject("LineGameObject");
                    if (only30Points)
                        lineGameObject.transform.SetParent(gameObjects[50].transform);
                    else
                        lineGameObject.transform.SetParent(gameObjects[48].transform);
                    lineGameObject.layer = gameObject.layer;
                    lineRenderers[i] = lineGameObject.AddComponent(typeof(LineRenderer)) as LineRenderer;
                } else if (i == 67) {
                    GameObject lineGameObject = new GameObject("LineGameObject");
                    if (only30Points)
                        lineGameObject.transform.SetParent(gameObjects[55].transform);
                    else
                        lineGameObject.transform.SetParent(gameObjects[53].transform);
                    lineGameObject.layer = gameObject.layer;
                    lineRenderers[i] = lineGameObject.AddComponent(typeof(LineRenderer)) as LineRenderer;
                }
                else
                    lineRenderers[i] = gameObjects[i].AddComponent(typeof(LineRenderer)) as LineRenderer;
                lineRenderers[i].useWorldSpace = true;
                lineRenderers[i].generateLightingData = true;
                lineRenderers[i].material = lineMaterial;
                lineRenderers[i].receiveShadows = receiveShadows;
                lineRenderers[i].widthMultiplier = lineWidth;
                lineRenderers[i].positionCount = 2;
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
        if (openSeeData == null || (show3DPoints && openSeeData.fit3DError > openSee.maxFit3DError))
            return;
        if (openSeeData.time > updated) {
            updated = openSeeData.time;
        } else {
            return;
        }
        if (show3DPoints) {
            centerBall.gameObject.SetActive(false);
            for (int i = 0; i < total; i++) {
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
                renderer.receiveShadows = receiveShadows;
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
        for (int i = 0; i < 68; i++) {
            if ((!only30Points && lines[i] == -1) || (only30Points && linesPt30Set[i] == -1))
                continue;
            if (!showLines || lineMaterial == null) {
                lineRenderers[i].enabled = false;
            } else {
                int a = i;
                int b = lines[i];
                if (only30Points) {
                    b = linesPt30Set[i];
                    if (i == 66)
                        a = 50;
                    if (i == 67)
                        a = 55;
                } else {
                    if (i == 66)
                        a = 48;
                    if (i == 67)
                        a = 53;
                }
                Color color = Color.Lerp(Color.red, Color.green, Mathf.Lerp(0.5f, openSeeData.confidence[a], openSeeData.confidence[b]));
                lineRenderers[i].enabled = true;
                lineRenderers[i].widthMultiplier = lineWidth;
                lineRenderers[i].receiveShadows = receiveShadows;
                lineRenderers[i].material.SetColor("_Color", color);
                lineRenderers[i].startColor = color;
                lineRenderers[i].endColor = color;
                lineRenderers[i].SetPosition(0, gameObjects[a].transform.position);
                lineRenderers[i].SetPosition(1, gameObjects[b].transform.position);
            }
        }
	}
}

}