using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using UnityEngine;

namespace OpenSee {

public class SVMModel {
	#region DllImport
	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, EntryPoint = "trainModel")]
	private static extern System.IntPtr trainModelDll(float[] features, float[] labels, float[] weights, int rows, int cols, int classes, int probability, float C);
    protected virtual System.IntPtr trainModel(float[] features, float[] labels, float[] weights, int rows, int cols, int classes, int probability, float C) {
        Debug.Log("Training LibSVM");
        return trainModelDll(features, labels, weights, rows, cols, classes, probability, C);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, EntryPoint = "predict")]
	private static extern void predictDll(System.IntPtr model, float[] features, [Out] float[] predictions, [Out] double[] probabilities, int rows);
    protected virtual void predict(System.IntPtr model, float[] features, [Out] float[] predictions, [Out] double[] probabilities, int rows) {
        predictDll(model, features, predictions, probabilities, rows);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "loadModel")]
	private static extern System.IntPtr loadModelDll([MarshalAs(UnmanagedType.LPStr)]string filename, int cols, int classes, double[] means, double[] sdevs);
    protected virtual System.IntPtr loadModel([MarshalAs(UnmanagedType.LPStr)]string filename, int cols, int classes, double[] means, double[] sdevs) {
        return loadModelDll(filename, cols, classes, means, sdevs);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "loadModelString")]
	private static extern System.IntPtr loadModelStringDll([MarshalAs(UnmanagedType.LPStr)]string modelString, int cols, int classes, double[] means, double[] sdevs);
    protected virtual System.IntPtr loadModelString([MarshalAs(UnmanagedType.LPStr)]string modelString, int cols, int classes, double[] means, double[] sdevs) {
        Debug.Log("Loading LibSVM");
        return loadModelStringDll(modelString, cols, classes, means, sdevs);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getScales")]
	private static extern void getScalesDll(System.IntPtr model, [Out] double[] means, [Out] double[] sdevs);
    protected virtual void getScales(System.IntPtr model, [Out] double[] means, [Out] double[] sdevs) {
        getScalesDll(model, means, sdevs);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "saveModel")]
	private static extern int saveModelDll(System.IntPtr model, [MarshalAs(UnmanagedType.LPStr)]string filename);
    protected virtual int saveModel(System.IntPtr model, [MarshalAs(UnmanagedType.LPStr)]string filename) {
        return saveModelDll(model, filename);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "saveModelString")]
	private static extern string saveModelStringDll(System.IntPtr model);
    protected virtual string saveModelString(System.IntPtr model) {
        return saveModelStringDll(model);
    }

	[DllImport("SVMModel", CallingConvention = CallingConvention.Cdecl, EntryPoint = "destroyModel")]
	private static extern void destroyModelDll(System.IntPtr model);
    protected virtual void destroyModel(System.IntPtr model) {
        destroyModelDll(model);
    }
	#endregion

    private bool haveModel = false;
    private System.IntPtr model;
    private int cols;
    private int maxClasses;

    [Serializable]
    private class SVMModelRepresentation {
        public string modelString;
        public int cols;
        public int maxClasses;
        public double[] means;
        public double[] sdevs;
    }

    public System.IntPtr LoadSerialized(byte[] modelBytes, out int cols, out int maxClasses, bool compress) {
        IFormatter formatter = new BinaryFormatter();
        MemoryStream memoryStream = new MemoryStream(modelBytes);
        memoryStream.Position = 0;
        SVMModelRepresentation smr;
        if (compress) {
            using (GZipStream gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress)) {
                smr = formatter.Deserialize(gzipStream) as SVMModelRepresentation;
            }
        } else {
            smr = formatter.Deserialize(memoryStream) as SVMModelRepresentation;
        }
        cols = smr.cols;
        maxClasses = smr.maxClasses;
        System.IntPtr model = loadModelString(smr.modelString, smr.cols, smr.maxClasses, smr.means, smr.sdevs);
        return model;
    }

    public byte[] ToSerialized(System.IntPtr model, int cols, int maxClasses, bool compress) {
        SVMModelRepresentation smr = new SVMModelRepresentation();
        smr.cols = cols;
        smr.maxClasses = maxClasses;
        smr.means = new double[cols];
        smr.sdevs = new double[cols];
        getScales(model, smr.means, smr.sdevs);
        smr.modelString = saveModelString(model);

        IFormatter formatter = new BinaryFormatter();
        MemoryStream memoryStream = new MemoryStream();
        if (compress) {
            using (GZipStream gzipStream = new GZipStream(memoryStream, CompressionMode.Compress)) {
                formatter.Serialize(gzipStream, smr);
                gzipStream.Flush();
            }
        } else {
            formatter.Serialize(memoryStream, smr);
        }
        return memoryStream.ToArray();
    }

    public void DestroyModel() {
        if (haveModel) {
            destroyModel(model);
            haveModel = false;
        }
    }

    private void LoadModel(byte[] modelBytes, bool compress) {
        DestroyModel();
        model = LoadSerialized(modelBytes, out cols, out maxClasses, compress);
        haveModel = true;
    }

    public void LoadModel(byte[] modelBytes) {
        LoadModel(modelBytes, true);
    }

    private byte[] SaveModel(bool compress) {
        if (!haveModel)
            return null;
        return ToSerialized(model, cols, maxClasses, compress);
    }

    public byte[] SaveModel() {
        return SaveModel(true);
    }

    // features is a rows*cols long float array with rows written one after another. labels is a rows long float array with the corresponding classes as integral numbers from 0 to num_classes-1, where all classes have to exist. Training on more than 10000 rows is disabled.
    public bool TrainModel(float[] features, float[] labels, float[] weights, int rows, int cols, int probability, float C) {
        DestroyModel();
        //if (rows > 10000)
        //    return false;
        if (cols < 1 || rows < 1 || features.Length < rows * cols || labels.Length < rows)
            return false;

        int max = -1;
        HashSet<int> knownClasses = new HashSet<int>();
        for (int i = 0; i < rows; i++) {
            labels[i] = (float)Math.Round(labels[i]);
            int l = (int)labels[i];
            if (l > max)
                max = l;
            if (l < 0)
                return false;
            knownClasses.Add(l);
        }
        for (int i = 0; i <= max; i++) {
            if (!knownClasses.Contains(i))
                return false;
        }

        maxClasses = max + 1;
        this.cols = cols;
        float start = Time.realtimeSinceStartup;
        model = trainModel(features, labels, weights, rows, cols, maxClasses, probability, C);
        Debug.Log("Training took: " + (Time.realtimeSinceStartup - start).ToString());
        haveModel = true;

        return true;
    }

    // This function takes the same arguments as train, except for cols, which is not necessary. The diagonal are correctly classified entries. The first index is labels, the second index is classifications.
    public int[,] ConfusionMatrix(float[] features, float[] labels, int rows, out float accuracy) {
        accuracy = 0f;
        if (!haveModel)
            return null;
        if (rows < 1 || features.Length < rows * cols || labels.Length < rows)
            return null;
        for (int i = 0; i < rows; i++) {
            labels[i] = (float)Math.Round(labels[i]);
            if (labels[i] < 0 || labels[i] >= maxClasses)
                return null;
        }
        int[,] confusionMatrix = new int[maxClasses, maxClasses];
        float[] predictions = new float[rows];
        double[] probabilities = new double[rows * maxClasses];
        predict(model, features, predictions, probabilities, rows);
        for (int i = 0; i < rows; i++) {
            predictions[i] = (float)Math.Round(predictions[i]);
            if (predictions[i] >= 0 && predictions[i] < maxClasses) {
                confusionMatrix[(int)labels[i], (int)predictions[i]]++;
                if ((int)labels[i] == (int)predictions[i])
                    accuracy++;
            }
        }
        accuracy /= (float)rows;
        return confusionMatrix;
    }

    static public string FormatMatrix(int[,] matrix) {
        return FormatMatrix(matrix, null);
    }

    static public string FormatMatrix(int[,] matrix, string[] labels) {
        StringBuilder sb = new StringBuilder();
        int rowLength = matrix.GetLength(0);
        int colLength = matrix.GetLength(1);
        int maxLableLen = 5;

        if (labels != null) {
            if (labels.Length != rowLength || labels.Length != colLength)
                labels = null;
            else {
                foreach (string label in labels)
                    if (label.Length > maxLableLen)
                        maxLableLen = label.Length;
                sb.Append("pred:".PadLeft(maxLableLen));
                sb.Append("     ");
                foreach (string label in labels) {
                    sb.Append(label.PadLeft(maxLableLen));
                    sb.Append("     ");
                }
                sb.Append(Environment.NewLine);
            }
        }

        for (int r = 0; r < rowLength; r++) {
            if (labels != null) {
                sb.Append(labels[r].PadLeft(maxLableLen));
                sb.Append("     ");
            }
            for (int c = 0; c < colLength; c++) {
                string n = matrix[r,c].ToString();
                sb.Append(n.PadLeft(maxLableLen));
                sb.Append("     ");
            }
            sb.Append(Environment.NewLine);
        }
        return sb.ToString();
    }

    // features is a rows*cols long float array with rows written one after another.
    public float[] Predict(float[] features, out double[] probabilities, int rows) {
        probabilities = null;
        if (!haveModel)
            return null;
        if (rows < 1 || features.Length != rows * cols)
            return null;
        float[] predictions = new float[rows];
        probabilities = new double[rows * maxClasses];
        predict(model, features, predictions, probabilities, rows);
        return predictions;
    }

    public float[] Predict(float[] features, int rows) {
        double[] probabilities;
        return Predict(features, out probabilities, rows);
    }

    public bool Ready() {
        return haveModel;
    }

    public SVMModel() {
    }

    public SVMModel(byte[] modelBytes) {
        LoadModel(modelBytes);
    }

    ~SVMModel() {
        DestroyModel();
    }
}

}