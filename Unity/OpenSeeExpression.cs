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
    [Tooltip("When enabled ThunderSVM is used instead of LibSVM, which is usually faster, but may cause issues on some systems.")]
    public bool useThunderSVM = false;
    [Header("Simple mode")]
    [Tooltip("When enabled, some simple heuristics will be used to detect expressions, rather than an SVM model. Outputs one of: neutral, fun, angry, surprise")]
    public bool simpleMode = false;
    [Tooltip("This smoothing factor is applied to the features used for simple expression detection.")]
    [Range(0f, 1f)]
    public float simpleSmoothing = 0.6f;
    [Tooltip("This smoothing factor is applied to the features used for simple expression detection.")]
    [Range(0f, 5f)]
    public float simpleSensitivity = 1f;
    [Header("Calibration")]
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
    [Tooltip("When enabled, training will be done in such a way that the model can output probabilities. This is a lot slower!")]
    public bool enableProbabilityTraining = false;
    [Tooltip("This is the SVM training C value.")]
    public float C = 2f;
    [Tooltip("This map allows you to set weights for different classes.")]
    public Dictionary<string, float> weightMap = null;
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
    [Header("Point selection")]
    [Tooltip("These settings can be used to train an expression detection model on a subset of points.")]
    public PointSelection pointSelection;
    [Header("Information")]
    [Tooltip("This is the number of expressions for which calibration data was collected. The maximum number is 10.")]
    public int expressionNumber = 0;
    [Tooltip("This is the percentage of necessary training data collected for the current expression.")]
    public float percentRecorded = 0.0f;
    [Tooltip("This shows whether a model is ready to predict expressions with.")]
    public bool modelReady = false;
    [Tooltip("This is the time the current expression was detected.")]
    public float expressionTime = 0f;
    [Tooltip("This is currently detected expression.")]
    public string expression = "";
    [Tooltip("These are the probabilities of all expressions. The labels in order can be retrieved through the GetClassLabels methods.")]
    public double[] probabilities;
    [Header("Training statistics")]
    [Tooltip("This shows the accuracy result of the last training run.")]
    public float accuracy = 0.0f;
    [Tooltip("This is the confusion matrix for the last training run's validation set.")]
    public int[,] confusionMatrix = null;
    [Tooltip("This is a pretty printed string representation of the confusion matrix.")]
    public string confusionMatrixString = "";
    [Tooltip("This shows any accuracy warnings that resulted from the last training run.")]
    public string[] warnings = null;
    
    private float lastMouthCorner = 0f;
    private float lastEyebrows = 0f;
    private bool hadFun = false;
    private bool hadAngry = false;
    private bool hadSurprised = false;
    float AdjustThreshold(bool active) {
        if (active)
            return 0.8f;
        else
            return 1f;
    }
    void ThresholdDetection() {
        lastMouthCorner = lastMouthCorner * simpleSmoothing + (openSeeData.features.MouthCornerUpDownLeft + openSeeData.features.MouthCornerUpDownRight) * 0.5f * (1f - simpleSmoothing);
        lastEyebrows = lastEyebrows * simpleSmoothing + (openSeeData.features.EyebrowUpDownLeft + openSeeData.features.EyebrowUpDownRight) * 0.5f * (1f - simpleSmoothing);
        if (lastMouthCorner * simpleSensitivity < -0.2f * AdjustThreshold(hadFun)) {
            expression = "fun";
            hadFun = true;
            hadSurprised = false;
            hadAngry = false;
        } else if (lastEyebrows * simpleSensitivity > 0.2f * AdjustThreshold(hadSurprised)) {
            expression = "surprise";
            hadFun = false;
            hadSurprised = true;
            hadAngry = false;
        } else if (lastEyebrows * simpleSensitivity < -0.25f * AdjustThreshold(hadAngry) && lastMouthCorner * simpleSensitivity > -0.3f * (2f - AdjustThreshold(hadAngry))) {
            expression = "angry";
            hadFun = false;
            hadSurprised = false;
            hadAngry = true;
        } else {
            expression = "neutral";
            hadFun = false;
            hadSurprised = false;
            hadAngry = false;
        }
        expressionTime = Time.time;
    }

    [Serializable]
    public class PointSelection {
        [Tooltip("When enabled, the points of the face contour will be used to determine the expression.")]
        public bool pointsFaceContour = true;
        [Tooltip("When enabled, the points of the right brow will be used to determine the expression.")]
        public bool pointsBrowRight = true;
        [Tooltip("When enabled, the points of the left brow will be used to determine the expression.")]
        public bool pointsBrowLeft = true;
        [Tooltip("When enabled, the points of the right eye will be used to determine the expression.")]
        public bool pointsEyeRight = true;
        [Tooltip("When enabled, the points of the left eye will be used to determine the expression.")]
        public bool pointsEyeLeft = true;
        [Tooltip("When enabled, the points of the nose will be used to determine the expression.")]
        public bool pointsNose = true;
        [Tooltip("When enabled, the points of the corners of the mouth will be used to determine the expression.")]
        public bool pointsMouthCorner = true;
        [Tooltip("When enabled, the points of the upper lip will be used to determine the expression.")]
        public bool pointsLipUpper = true;
        [Tooltip("When enabled, the points of the lower lip will be used to determine the expression.")]
        public bool pointsLipLower = true;
        [Tooltip("When enabled, certain manually designed features will be used to determine the expression.")]
        public bool features = true;
        [Tooltip("When enabled, the depth value of points is also used for training.")]
        public bool includeDepth = false;
    }
    [Serializable]
    private class OpenSeeExpressionRepresentation {
        private Dictionary<string, List<float[]>> expressions;
        private byte[] modelBytes = null;
        private string[] classLabels = null;
        private int[] indices = null;
        private PointSelection pointSelection;
        private bool thunderSVM = false;
        //private bool newModel = false;

        static public void LoadSerialized(byte[] modelBytes, out Dictionary<string, List<float[]>> expressions, out SVMModel model, out string[] classLabels, out int[] indices, ref PointSelection pointSelection) {
            IFormatter formatter = new BinaryFormatter();
            MemoryStream memoryStream = new MemoryStream(modelBytes);
            memoryStream.Position = 0;
            OpenSeeExpressionRepresentation oser;
            using (GZipStream gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress)) {
                oser = formatter.Deserialize(gzipStream) as OpenSeeExpressionRepresentation;
            }
            expressions = oser.expressions;
            /*if (oser.thunderSVM)
                model = new ThunderSVMModel(oser.modelBytes);
            else*/
            model = new SVMModel(oser.modelBytes);
            classLabels = oser.classLabels;
            indices = oser.indices;
            //pointSelection = oser.pointSelection;
            if (indices == null) {
                indices = new int[1 + 1 + 3 + 4 + 3 + 3 * 66];
                for (int i = 0; i < 1 + 1 + 3 + 4 + 3 + 3 * 66; i++)
                    indices[i] = i;
            }
            /*if (pointSelection == null)
                pointSelection = new PointSelection();
            if (!oser.newModel)
                pointSelection.features = true;*/
        }

        static public byte[] ToSerialized(Dictionary<string, List<float[]>> expressions, SVMModel model, string[] classLabels, int[] indices, PointSelection pointSelection) {
            OpenSeeExpressionRepresentation oser = new OpenSeeExpressionRepresentation();
            oser.expressions = expressions;
            oser.modelBytes = model.SaveModel();
            oser.classLabels = classLabels;
            oser.indices = indices;
            oser.pointSelection = pointSelection;
            oser.thunderSVM = model is ThunderSVMModel;
            //oser.newModel = true;

            IFormatter formatter = new BinaryFormatter();
            MemoryStream memoryStream = new MemoryStream();
            using (GZipStream gzipStream = new GZipStream(memoryStream, CompressionMode.Compress)) {
                formatter.Serialize(gzipStream, oser);
                gzipStream.Flush();
            }
            return memoryStream.ToArray();
        }
    }

    private OpenSee.OpenSeeData openSeeData = null;
    private Dictionary<string, List<float[]>> expressions;
    private SVMModel model = null;
    private string[] classLabels = null;
    private int maxSamples = 1666;
    // rightEyeOpen, leftEyeOpen, translation, rawQuaternion, rawEuler, confidence, points/(width, height), points3D, features
    private int colsFull = 1 + 1 + 3 + 4 + 3 + /*66 + 2 * 66  +*/ 3 * 66 + 14;
    private int colsBase = 1 + 1 + 3 + 4 + 3;
    private int cols;
    private double lastCapture = 0.0;
    private float warningThreshold = 5;
    private int currentPrediction = -1;
    private int lastPrediction = -1;
    private int lastPredictionCount = 0;
    private int frameCount = 0;
    private System.Random rnd;

    private int[] indices;

    private int[] indicesFaceContour = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    private int[] indicesBrowRight = new int[] {17, 18, 19, 20, 21};
    private int[] indicesBrowLeft = new int[] {22, 23, 24, 25, 26};
    private int[] indicesEyeRight = new int[] {36, 37, 38, 39, 40, 41};
    private int[] indicesEyeLeft = new int[] {42, 43, 44, 45, 46, 47};
    private int[] indicesNose = new int[] {27, 28, 29, /*30, */31, 32, 33, 34, 35};
    private int[] indicesMouthCorner = new int[] {58, 62};
    private int[] indicesLipUpper = new int[] {48, 49, 50, 51, 52, 59, 60, 61};
    private int[] indicesLipLower = new int[] {53, 54, 55, 56, 57, 63, 64, 65};

    private void SelectPoints() {
        List<int> indexList = new List<int>();
        for (int i = 0; i < colsBase; i++)
            if (i > 4) // Translation and (sometimes unavailable) eye open fields should not be used for training!
                indexList.Add(i);
        cols = indexList.Count;
        if (pointSelection.pointsFaceContour)
            foreach (int i in indicesFaceContour) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsBrowRight)
            foreach (int i in indicesBrowRight) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsBrowLeft)
            foreach (int i in indicesBrowLeft) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsEyeRight)
            foreach (int i in indicesEyeRight) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsEyeLeft)
            foreach (int i in indicesEyeLeft) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsNose)
            foreach (int i in indicesNose) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsNose)
            foreach (int i in indicesMouthCorner) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsLipUpper)
            foreach (int i in indicesLipUpper) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.pointsLipLower)
            foreach (int i in indicesLipLower) {
                indexList.Add(colsBase + i * 3);
                indexList.Add(colsBase + i * 3 + 1);
                cols += 2;
                if (pointSelection.includeDepth) {
                    indexList.Add(colsBase + i * 3 + 2);
                    cols++;
                }
            }
        if (pointSelection.features)
            for (int i = 0; i < 14; i++) {
                indexList.Add(colsBase + 66 * 3 + i);
                cols++;
            }
        indices = indexList.ToArray();
    }

    private void ResetInfo() {
        recording = false;
        clear = false;
        train = false;
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
        if (useThunderSVM)
            model = new ThunderSVMModel();
        else
            model = new SVMModel();
        rnd = new System.Random();
    }

    private float[] GetData(OpenSee.OpenSeeData t) {
        float[] data = new float[colsFull];
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
        data[12 + 66 * 3 + 0] = t.features.EyeLeft;
        data[12 + 66 * 3 + 1] = t.features.EyeRight;
        data[12 + 66 * 3 + 2] = t.features.EyebrowSteepnessLeft;
        data[12 + 66 * 3 + 3] = t.features.EyebrowUpDownLeft;
        data[12 + 66 * 3 + 4] = t.features.EyebrowQuirkLeft;
        data[12 + 66 * 3 + 5] = t.features.EyebrowSteepnessRight;
        data[12 + 66 * 3 + 6] = t.features.EyebrowUpDownRight;
        data[12 + 66 * 3 + 7] = t.features.EyebrowQuirkRight;
        data[12 + 66 * 3 + 8] = t.features.MouthCornerUpDownLeft;
        data[12 + 66 * 3 + 9] = t.features.MouthCornerInOutLeft;
        data[12 + 66 * 3 + 10] = t.features.MouthCornerUpDownRight;
        data[12 + 66 * 3 + 11] = t.features.MouthCornerInOutRight;
        data[12 + 66 * 3 + 12] = t.features.MouthOpen;
        data[12 + 66 * 3 + 13] = t.features.MouthWide;
        return data;
    }

    public bool TrainModel() {
        Debug.Log("---------------------");
        train = false;
        if (openSee == null) {
            ResetInfo();
            return false;
        }
        modelReady = false;
        if (model is SVMModel && useThunderSVM)
            model = new ThunderSVMModel();
        else if (model is ThunderSVMModel && !useThunderSVM)
            model = new SVMModel();
        SelectPoints();
        List<string> keys = new List<string>();
        List<string> accuracyWarnings = new List<string>();
        int samples = 0;
        foreach (string key in expressions.Keys) {
            List<float[]> list = new List<float[]>(expressions[key]);
            list.RemoveAll(x => x.Length != colsFull);
            if (list != null && list.Count == maxSamples) {
                Debug.Log("[Training info] Adding expression " + key + " to training data.");
                keys.Add(key);
                samples += list.Count;
            } else {
                if (list != null && list.Count > 20) {
                    Debug.Log("[Training warning] Expression " + key + " has little data and might be inaccurate. (" + list.Count + ")");
                    accuracyWarnings.Add("Expression " + key + " has little data and might be inaccurate. (" + list.Count + ")");
                    samples += list.Count;
                    keys.Add(key);
                } else {
                    Debug.Log("[Training warning] Skipping expression " + key + " due to lack of collected data. (" + list.Count + ")");
                    accuracyWarnings.Add("Skipping expression " + key + " due to lack of collected data. (" + list.Count + ")");
                    continue;
                }
            }
        }
        int classes = keys.Count;
        if (classes < 2 || classes > 10) {
            Debug.Log("[Training error] The number of expressions that can be used for training is " + classes + ", which is either below 2 or higher than 10.");
            accuracyWarnings.Add("[Training error] The number of expressions that can be used for training is " + classes + ", which is either below 2 or higher than 10.");
            return false;
        }
        keys.Sort();
        classLabels = keys.ToArray();
        
        float[] weights = null;
        if (weightMap != null) {
            weights = new float[classLabels.Length];
            for (int i = 0; i < classLabels.Length; i++) {
                if (weightMap.ContainsKey(classLabels[i]))
                    weights[i] = weightMap[classLabels[i]];
                else
                    weights[i] = 1f;
                Debug.Log("[Training info] Adding weight " + weights[i] + " for " + classLabels[i] + ".");
            }
        }
        
        Debug.Log("[Training info] Preparing trainig data.");
        
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
            List<float[]> list = new List<float[]>(expressions[keys[i]]);
            list.RemoveAll(x => x.Length != colsFull);
            int local_train_split = list.Count * 3 / 4;
            test_split = list.Count - local_train_split;
            for (int j = list.Count; j > 1;) {
                j--;
                int k = rnd.Next(j + 1);
                float[] tmp = list[k];
                list[k] = list[j];
                list[j]= tmp;
            }
            int train_split_current = train_split;
            if (classLabels[i] == "neutral")
                train_split_current *= 1;
            for (int j = 0; j < train_split_current; j++) {
                float factor = 1f;
                float adder = 0f;
                /*if (j > train_split || j > list.Count) {
                    factor = UnityEngine.Random.Range(0.98f, 1.02f);
                    adder = UnityEngine.Random.Range(-0.02f, 0.02f);
                }*/
                for (int k = 0; k < cols; k++) {
                    float v = list[j % local_train_split][indices[k]] * factor + adder;
                    X_train[i_train * cols + k] = v;
                }
                y_train[i_train] = i;
                i_train++;
            }
            for (int j = local_train_split; j < local_train_split + test_split; j++) {
                for (int k = 0; k < cols; k++) {
                    float v = list[j][indices[k]];
                    X_test[i_test * cols + k] = v;
                }
                y_test[i_test] = i;
                i_test++;
            }
        }
        Debug.Log("[Training info] Preparation complete.");
        int probability = 0;
        if (enableProbabilityTraining)
            probability = 1;
        model.TrainModel(X_train, y_train, weights, i_train, cols, probability, C);
        confusionMatrix = model.ConfusionMatrix(X_test, y_test, i_test, out accuracy);
        confusionMatrixString = SVMModel.FormatMatrix(confusionMatrix, classLabels);
        for (int label = 0; label < classes; label++) {
            int total = 0;
            for (int p = 0; p < classes; p++) {
                total += confusionMatrix[label, p];
            }
            float error = 100f * (1f - ((float)confusionMatrix[label, label] / (float)total));
            if (error > warningThreshold) {
                accuracyWarnings.Add("[Training warning] The expression \"" + classLabels[label] + "\" is misclassified with a chance of " + error.ToString("0.00") + "%.");
                Debug.Log("[Training warning] The expression \"" + classLabels[label] + "\" is misclassified with a chance of " + error.ToString("0.00") + "%.");
            }
        }
        warnings = accuracyWarnings.ToArray();
        modelReady = true;
        Debug.Log("[Training info] Trained model.");
        return true;
    }

    public bool PredictExpression() {
        if (openSee == null)
            return false;
        if (!simpleMode && !(modelReady && model.Ready())) {
            ResetInfo();
            return false;
        }
        openSeeData = openSee.GetOpenSeeData(faceId);
        if (openSeeData == null || openSeeData.time <= lastCapture)
            return false;
        if (simpleMode) {
            ThresholdDetection();
            return true;
        }
        float[] faceData = GetData(openSeeData);
        float[] predictionData = new float[cols];
        for (int i = 0; i < cols; i++)
            predictionData[i] = faceData[indices[i]];
        float[] prediction = model.Predict(predictionData, out probabilities, 1);
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
        if (expressions.ContainsKey(calibrationExpression) || expressions.Keys.Count < 10) {
            openSeeData = openSee.GetOpenSeeData(faceId);
            if (openSeeData == null || openSeeData.time <= lastCapture)
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
                list[rnd.Next(maxSamples)] = GetData(openSeeData);
            } else
                list.Add(GetData(openSeeData));
            percentRecorded = 100f * (float)list.Count/(float)maxSamples;
            return true;
        } else {
            recording = false;
            calibrationExpression = "";
            return false;
        }
    }

    public string[] GetClassLabels() {
        List<string> labels = new List<string>();
        foreach (string l in classLabels)
            labels.Add(String.Copy(l));
        return labels.ToArray();
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
        OpenSeeExpressionRepresentation.LoadSerialized(data, out expressions, out model, out classLabels, out indices, ref pointSelection);
        cols = indices.Length;
        if (model.Ready())
            modelReady = true;
    }

    public byte[] SaveToBytes() {
        save = false;
        if (openSee == null) {
            ResetInfo();
            return null;
        }
        return OpenSeeExpressionRepresentation.ToSerialized(expressions, model, classLabels, indices, pointSelection);
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
        else if (simpleMode || modelReady)
            PredictExpression();
        else
            predict = false;

        if (expressions.ContainsKey(calibrationExpression))
            percentRecorded = 100f * (float)expressions[calibrationExpression].Count/(float)maxSamples;
        else
            percentRecorded = 0f;

        if (openSeeData == null)
            return;
       lastCapture = openSeeData.time;
   }
}

}