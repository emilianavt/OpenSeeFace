using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using OpenSee;
using UnityEngine;

public class ThunderSVMModel : SVMModel {
	#region DllImport
	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, EntryPoint = "trainModel")]
	private static extern System.IntPtr trainModelDll(float[] features, float[] labels, float[] weights, int rows, int cols, int classes, int probability, float C);
    protected override System.IntPtr trainModel(float[] features, float[] labels, float[] weights, int rows, int cols, int classes, int probability, float C) {
        Debug.Log("Training ThunderSVM");
        return trainModelDll(features, labels, weights, rows, cols, classes, probability, C);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, EntryPoint = "predict")]
	private static extern void predictDll(System.IntPtr model, float[] features, [Out] float[] predictions, [Out] double[] probabilities, int rows);
    protected override void predict(System.IntPtr model, float[] features, [Out] float[] predictions, [Out] double[] probabilities, int rows) {
        predictDll(model, features, predictions, probabilities, rows);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "loadModel")]
	private static extern System.IntPtr loadModelDll([MarshalAs(UnmanagedType.LPStr)]string filename, int cols, int classes, double[] means, double[] sdevs);
    protected override System.IntPtr loadModel([MarshalAs(UnmanagedType.LPStr)]string filename, int cols, int classes, double[] means, double[] sdevs) {
        return loadModelDll(filename, cols, classes, means, sdevs);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "loadModelString")]
	private static extern System.IntPtr loadModelStringDll([MarshalAs(UnmanagedType.LPStr)]string modelString, int cols, int classes, double[] means, double[] sdevs);
    protected override System.IntPtr loadModelString([MarshalAs(UnmanagedType.LPStr)]string modelString, int cols, int classes, double[] means, double[] sdevs) {
        Debug.Log("Loading ThunderSVM");
        return loadModelStringDll(modelString, cols, classes, means, sdevs);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, EntryPoint = "getScales")]
	private static extern void getScalesDll(System.IntPtr model, [Out] double[] means, [Out] double[] sdevs);
    protected override void getScales(System.IntPtr model, [Out] double[] means, [Out] double[] sdevs) {
        getScalesDll(model, means, sdevs);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "saveModel")]
	private static extern int saveModelDll(System.IntPtr model, [MarshalAs(UnmanagedType.LPStr)]string filename);
    protected override int saveModel(System.IntPtr model, [MarshalAs(UnmanagedType.LPStr)]string filename) {
        return saveModelDll(model, filename);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "saveModelString")]
	private static extern string saveModelStringDll(System.IntPtr model);
    protected override string saveModelString(System.IntPtr model) {
        return saveModelStringDll(model);
    }

	[DllImport("ThunderSVMWrapper", CallingConvention = CallingConvention.Cdecl, EntryPoint = "destroyModel")]
	private static extern void destroyModelDll(System.IntPtr model);
    protected override void destroyModel(System.IntPtr model) {
        destroyModelDll(model);
    }
	#endregion
    
    public ThunderSVMModel() {
    }

    public ThunderSVMModel(byte[] modelBytes) {
        LoadModel(modelBytes);
    }
}