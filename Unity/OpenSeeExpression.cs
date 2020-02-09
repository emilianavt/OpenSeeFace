using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

namespace OpenSee {

public class OpenSeeExpression : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("This is the source of face tracking data. If it is not set, any OpenSee component on the current object is used.")]
    public OpenSee openSee;
    [Tooltip("This specifies which expression calibration data should be collected for.")]
    public string calibrationExpression = "neutral";
    [Tooltip("This specifies the id of the face for which data should be collected and predictions should be made. Face ids depend only on the order of first detection and locations of the faces.")]
    public int faceId = 0;
    [Tooltip("This keeps low numbers of misdetections from changing the expression, but introduces a bit of lag to expression changes. For example, a value of 1 means that a new expression has to be detected twice in a row, before the predicted expression changes.")]
    public int expressionStabilizer = 1;
    [Tooltip("Setting this to a value above 1 will record only every recordingSkip-th frame when capture calibration data. This can be useful with cameras that have a high framerate leading to only a short time for capturing varied expression information. Usually at least 20-30 seconds are required to capture a good variety of angles and expression variation, so this should be set accordingly when using the recording flag. At 1, no frames are skipped.")]
    public int recordingSkip = 2;
    [Tooltip("If allowOverRecording is enabled and expression data is recorded beyond 100%, this is used instead of recordingSkip to avoid quickly replacing the existing data.")]
    public int overRecordingSkip = 5;
    [Tooltip("This is the filename used for loading and saving expression data using the load and save flags.")]
    public string filename = "";
    [Header("Toggles")]
    [Tooltip("When enabled, calibration data will be collected from the given OpenSee component.")]
    public bool recording = false;
    [Tooltip("When enabled, you can keep recording past 100%. When doing so, random frames in the already collected data will be overwritten with new data. To prevent quickly replacing all the collected data, overRecordingSkip is used instead of recordingSkip, which should be set to a higher value.")]
    public bool allowOverRecording = false;
    [Tooltip("When enabled, the calibration data collected for the current expression is cleared and this flag is set back to false.")]
    public bool clear = false;
    [Tooltip("When enabled, a new prediction model is trained and this flag is set back to false. Only expressions for which percentRecorded is 100% will be included in training.")]
    public bool train = false;
    [Tooltip("When enabled and a model is loaded, the current expression will be predicted every frame.")]
    public bool predict = false;
    [Tooltip("When enabled, the expression data from the specified filename will be loaded and this flag is set back to false.")]
    public bool load = false;
    [Tooltip("When enabled, the expression data will be saved to the specified filename and this flag is set back to false.")]
    public bool save = false;
    [Header("Information")]
    [Tooltip("This is the of expressions for which calibration data was collected. The maximum number is 25.")]
    public int expressionNumber = 0;
    [Tooltip("This is the percentage of necessary training data collected for the current expression.")]
    public float percentRecorded = 0.0f;
    [Tooltip("This shows whether a model is ready to predict expressions with.")]
    public bool modelReady = false;
    [Tooltip("This is the time the current expression was detected.")]
    public float expressionTime = 0f;
    [Tooltip("This is currently detected expression.")]
    public string expression = "";
    [Header("Training statistics")]
    [Tooltip("This shows the accuracy result of the last training run.")]
    public float accuracy = 0.0f;
    [Tooltip("This is the confusion matrix for the last training run's validation set.")]
    public int[,] confusionMatrix = null;
    [Tooltip("This is a pretty printed string representation of the confusion matrix.")]
    public string confusionMatrixString = "";
    [Tooltip("This shows any accuracy warnings that resulted from the last training run.")]
    public string[] warnings = null;

    [Serializable]
    private class OpenSeeExpressionRepresentation {
        private Dictionary<string, List<float[]>> expressions;
        private byte[] modelBytes = null;
        private string[] classLabels = null;

        static public void LoadSerialized(byte[] modelBytes, out Dictionary<string, List<float[]>> expressions, out SVMModel model, out string[] classLabels) {
            IFormatter formatter = new BinaryFormatter();
            MemoryStream memoryStream = new MemoryStream(modelBytes);
            memoryStream.Position = 0;
            OpenSeeExpressionRepresentation oser;
            using (GZipStream gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress)) {
                oser = formatter.Deserialize(gzipStream) as OpenSeeExpressionRepresentation;
            }
            expressions = oser.expressions;
            model = new SVMModel(oser.modelBytes);
            classLabels = oser.classLabels;
        }

        static public byte[] ToSerialized(Dictionary<string, List<float[]>> expressions, SVMModel model, string[] classLabels) {
            OpenSeeExpressionRepresentation oser = new OpenSeeExpressionRepresentation();
            oser.expressions = expressions;
            oser.modelBytes = model.SaveModel();
            oser.classLabels = classLabels;

            IFormatter formatter = new BinaryFormatter();
            MemoryStream memoryStream = new MemoryStream();
            using (GZipStream gzipStream = new GZipStream(memoryStream, CompressionMode.Compress)) {
                formatter.Serialize(gzipStream, oser);
                gzipStream.Flush();
            }
            return memoryStream.ToArray();
        }
    }

    private Dictionary<string, List<float[]>> expressions;
    private SVMModel model = null;
    private string[] classLabels = null;
    private int maxSamples = 400; // Allows 25 expressions without exceeding 10000 rows in total for SVM training, which is a commonly seen upper limit.
    // rightEyeOpen, leftEyeOpen, translation, rawQuaternion, rawEuler, confidence, points/(width, height), points3D
    private int cols = 1 + 1 + 3 + 4 + 3 + /*66 + 2 * 66  +*/ 3 * 66;
    private double lastCapture = 0.0;
    private float warningThreshold = 5;
    private int currentPrediction = -1;
    private int lastPrediction = -1;
    private int lastPredictionCount = 0;
    private int frameCount = 0;
    private System.Random rnd;

    private void ResetInfo() {
        recording = false;
        clear = false;
        train = false;
        predict = false;
        modelReady = false;
        accuracy = 0.0f;
        confusionMatrix = null;
        warnings = null;
        percentRecorded = 0.0f;
        expression = "";
        confusionMatrixString = "";
        lastPrediction = -1;
        currentPrediction = -1;
        lastPredictionCount = 0;
        expressionTime = 0f;
    }

    void Start()
    {
        if (openSee == null)
            openSee = GetComponent<OpenSee>();
        if (openSee == null) {
            ResetInfo();
            return;
        }
        ResetInfo();
        expressions = new Dictionary<string, List<float[]>>();
        model = new SVMModel();
        rnd = new System.Random();
    }

    private float[] GetData(OpenSee.OpenSeeData t) {
        float[] data = new float[cols];
        data[0] = t.rightEyeOpen;
        data[1] = t.leftEyeOpen;
        data[2] = t.translation.x;
        data[3] = t.translation.y;
        data[4] = t.translation.z;
        data[5] = t.rawQuaternion.x;
        data[6] = t.rawQuaternion.y;
        data[7] = t.rawQuaternion.z;
        data[8] = t.rawQuaternion.w;
        data[9] = t.rawEuler.x;
        data[10] = t.rawEuler.y;
        data[11] = t.rawEuler.z;
        /*for (int i = 0; i < 66; i++)
            data[12 + i] = t.confidence[i];
        for (int i = 0; i < 66; i++) {
            data[12 + 66 + 2 * i] = t.points[i].x / t.cameraResolution.x;
            data[12 + 66 + 2 * i + 1] = t.points[i].y / t.cameraResolution.y;
        }*/
        for (int i = 0; i < 66; i++) {
            data[12 + /*66 + 2 * 66 +*/ 3 * i] = t.points3D[i].x;
            data[12 + /*66 + 2 * 66 +*/ 3 * i + 1] = t.points3D[i].y;
            data[12 + /*66 + 2 * 66 +*/ 3 * i + 2] = t.points3D[i].z;
        }
        return data;
    }

    public bool TrainModel() {
        train = false;
        if (openSee == null) {
            ResetInfo();
            return false;
        }
        List<string> keys = new List<string>();
        foreach (string key in expressions.Keys) {
            List<float[]> list = expressions[key];
            if (list != null && list.Count == maxSamples)
                keys.Add(key);
            else {
                Debug.Log("[Training warning] Skipping expression " + key + " due to lack of collected data.");
                continue;
            }
        }
        int classes = keys.Count;
        if (classes < 2 || classes > 25) {
            Debug.Log("[Training error] The number of expressions that can be used for training is " + classes + ", which is either below 2 or higher than 25.");
            return false;
        }
        keys.Sort();
        classLabels = keys.ToArray();
        int train_split = maxSamples * 3 / 4;
        int test_split = maxSamples - train_split;
        int rows_train = classes * train_split;
        int rows_test = classes * test_split;
        float[] X_train = new float[rows_train * cols];
        float[] X_test = new float[rows_test * cols];
        float[] y_train = new float[rows_train];
        float[] y_test = new float[rows_test];
        int i_train = 0;
        int i_test = 0;
        System.Random rnd = new System.Random();
        for (int i = 0; i < classes; i++) {
            List<float[]> list = expressions[keys[i]];
            for (int j = maxSamples; j > 1;) {
                j--;
                int k = rnd.Next(j + 1);
                float[] tmp = list[k];
                list[k] = list[j];
                list[j]= tmp;
            }
            for (int j = 0; j < train_split; j++) {
                for (int k = 0; k < cols; k++) {
                    X_train[i_train * cols + k] = list[j][k];
                }
                y_train[i_train] = i;
                i_train++;
            }
            for (int j = train_split; j < train_split + test_split; j++) {
                for (int k = 0; k < cols; k++)
                    X_test[i_test * cols + k] = list[j][k];
                y_test[i_test] = i;
                i_test++;
            }
        }
        model.TrainModel(X_train, y_train, rows_train, cols);
        confusionMatrix = model.ConfusionMatrix(X_test, y_test, rows_test, out accuracy);
        confusionMatrixString = SVMModel.FormatMatrix(confusionMatrix, classLabels);
        List<string> accuracyWarnings = new List<string>();
        for (int label = 0; label < classes; label++) {
            float error = 100f * (1f - ((float)confusionMatrix[label, label] / (float)test_split));
            if (error > warningThreshold)
                accuracyWarnings.Add("The expression \"" + classLabels[label] + "\" is misclassified with a chance of " + error.ToString("0.00") + "%.");
        }
        warnings = accuracyWarnings.ToArray();
        modelReady = true;
        return true;
    }
    
    public bool PredictExpression() {
        if (openSee == null || !(modelReady && model.Ready())) {
            ResetInfo();
            return false;
        }
        OpenSee.OpenSeeData[] openSeeData = openSee.trackingData;
        if (openSeeData == null)
            return false;
        OpenSee.OpenSeeData t = null;
        foreach (OpenSee.OpenSeeData data in openSeeData)
            if (data.id == faceId)
                t = data;
        if (t == null || t.time <= lastCapture)
            return false;
        float[] faceData = GetData(t);
        float[] prediction = model.Predict(faceData, 1);
        int predictedExpression = (int)Mathf.Round(prediction[0]);
        if (predictedExpression == lastPrediction)
            lastPredictionCount++;
        else {
            lastPrediction = predictedExpression;
            lastPredictionCount = 1;
        }
        if (lastPredictionCount > expressionStabilizer) {
            currentPrediction = lastPrediction;
            expressionTime = Time.time;
        }
        if (currentPrediction >= 0 && currentPrediction <= classLabels.Length)
            expression = classLabels[currentPrediction];
        else
            expression = "";
        return true;
    }

    public bool CaptureCalibrationData() {
        if (openSee == null) {
            ResetInfo();
            return false;
        }
        if (recordingSkip < 1)
            recordingSkip = 1;
        if (calibrationExpression == "")
            return false;
        if (expressions.ContainsKey(calibrationExpression) || expressions.Keys.Count < 25) {
            OpenSee.OpenSeeData[] openSeeData = openSee.trackingData;
            if (openSeeData == null)
                return false;
            OpenSee.OpenSeeData t = null;
            foreach (OpenSee.OpenSeeData data in openSeeData)
                if (data.id == faceId)
                    t = data;
            if (t == null || t.time <= lastCapture)
                return false;
            if (!expressions.ContainsKey(calibrationExpression)) {
                expressions.Add(calibrationExpression, new List<float[]>());
            }
            List<float[]> list = expressions[calibrationExpression];
            if (list.Count >= maxSamples) {
                percentRecorded = 100f;
                recording = allowOverRecording;
                if (!recording)
                    return false;
            }
            int skip = recordingSkip;
            if (percentRecorded >= 100f)
                skip = overRecordingSkip;
            if (frameCount % skip != 0)
                return true;
            if (percentRecorded >= 100f) {
                list[rnd.Next(maxSamples)] = GetData(t);
            } else
                list.Add(GetData(t));
            percentRecorded = 100f * (float)list.Count/(float)maxSamples;
            return true;
        } else {
            recording = false;
            calibrationExpression = "";
            return false;
        }
    }

    public string[] GetExpressions() {
        List<string> keys = new List<string>();
        foreach (string key in expressions.Keys)
            keys.Add(key);
        keys.Sort();
        return keys.ToArray();
    }
    
    public void ClearExpression() {
            if (expressions.ContainsKey(calibrationExpression))
                expressions.Remove(calibrationExpression);
            clear = false;
    }
    
    public void LoadFromBytes(byte[] data) {
        load = false;
        if (openSee == null) {
            ResetInfo();
            return;
        }
        OpenSeeExpressionRepresentation.LoadSerialized(data, out expressions, out model, out classLabels);
        if (model.Ready())
            modelReady = true;
    }
    
    public byte[] SaveToBytes() {
        save = false;
        if (openSee == null) {
            ResetInfo();
            return null;
        }
        return OpenSeeExpressionRepresentation.ToSerialized(expressions, model, classLabels);
    }
    
    public void LoadFromFile() {
        load = false;
        if (openSee == null) {
            ResetInfo();
            return;
        }
        byte[] data = File.ReadAllBytes(filename);
        LoadFromBytes(data);
    }
    
    public void SaveToFile() {
        save = false;
        if (openSee == null) {
            ResetInfo();
            return;
        }
        byte[] data = SaveToBytes();
        File.WriteAllBytes(filename, data);
    }

    void Update()
    {
        frameCount++;
        if (openSee == null || expressions == null) {
            ResetInfo();
            return;
        }
        if (clear)
            ClearExpression();
        if (recording && calibrationExpression != "") {
            CaptureCalibrationData();
        }
        if (train)
            TrainModel();
        expressionNumber = expressions.Keys.Count;
        if (load)
            LoadFromFile();
        if (save)
            SaveToFile();
        if (!predict)
            expression = "";
        else if (modelReady)
            PredictExpression();
        else
            predict = false;
        
        if (expressions.ContainsKey(calibrationExpression))
            percentRecorded = 100f * (float)expressions[calibrationExpression].Count/(float)maxSamples;
        else
            percentRecorded = 0f;
        
        OpenSee.OpenSeeData[] openSeeData = openSee.trackingData;
        if (openSeeData == null)
            return;
        OpenSee.OpenSeeData t = null;
        foreach (OpenSee.OpenSeeData data in openSeeData)
            if (data.id == faceId)
                t = data;
        if (t == null || t.time <= lastCapture)
            return;
       lastCapture = t.time;
   }
}

}