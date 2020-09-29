using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// To use this, add the following entry to the dependencies in Packages\manifest.json:
//       "com.unity.nuget.newtonsoft-json": "2.0.0"
using Newtonsoft.Json;

namespace OpenSee {

[Serializable]
public enum OpenSeeWebcamType { Unknown = -1, DirectShow, Blackmagic };

[Serializable]
public enum OpenSeeWebcamFormat { Any = 0, Unknown = 1, ARGB = 100, XRGB, I420 = 200, NV12, YV12, Y800, YVYU = 300, YUY2, UYVY, HDYC, MJPEG = 400, H264 };

[Serializable]
public struct OpenSeeWebcamCapability {
    public int id;
    
    // DirectShow
    public int minCX;
    public int minCY;
    public int maxCX;
    public int maxCY;
    public int granularityCX;
    public int granularityCY;
    public int minInterval;
    public int maxInterval;
    public int rating;
    public OpenSeeWebcamFormat format;
    
    // Blackmagic
    public int bmTimescale;
    public int bmFrameduration;
    public int bmModecode;
}

[Serializable]
public class OpenSeeWebcam {
    public OpenSeeWebcamType type = OpenSeeWebcamType.Unknown;
    public int id;
    public string name;

    // DirectShow
    public string path;
    
    // Blackmagic
    public long bmId;
    public int bmFlags;

    // General
    public OpenSeeWebcamCapability[] caps;
    
    public List<string> prettyCaps = null;
    
    private List<Tuple<OpenSeeWebcamCapability, int>> splitCaps = null;

    private int CompareCaps(OpenSeeWebcamCapability a, OpenSeeWebcamCapability b) {
        if (a.minCX * a.minCY > b.minCX * b.minCY)
            return -1;
        if (a.minCX * a.minCY < b.minCX * b.minCY)
            return 1;
        if (a.rating < b.rating)
            return -1;
        if (a.rating > b.rating)
            return 1;
        if (a.minInterval < b.minInterval)
            return -1;
        if (a.minInterval > b.minInterval)
            return 1;
        return 0;
    }
    
    private string GetPrettyCapability(OpenSeeWebcamCapability cap) {
        float fps = 10000000f / (float)cap.minInterval;
        string prettyCap = cap.minCX + "x" + cap.minCY + ", " + fps.ToString("0.##") + "fps (" + cap.format.ToString() + ")";
        return prettyCap;
    }
    
    public List<string> GetPrettyCapabilities() {
        if (splitCaps != null && prettyCaps != null)
            return prettyCaps;
        
        splitCaps = new List<Tuple<OpenSeeWebcamCapability, int>>();
        
        for (int i = 0; i < caps.Length; i++) {
            if (caps[i].minCX == caps[i].maxCX && caps[i].minCY == caps[i].maxCY)
                splitCaps.Add(new Tuple<OpenSeeWebcamCapability, int>(caps[i], i));
            else {
                OpenSeeWebcamCapability min = caps[i];
                OpenSeeWebcamCapability max = caps[i];
                min.maxCX = min.minCX;
                min.maxCY = min.minCY;
                max.minCX = max.maxCX;
                max.minCY = max.maxCY;
                splitCaps.Add(new Tuple<OpenSeeWebcamCapability, int>(min, i));
                splitCaps.Add(new Tuple<OpenSeeWebcamCapability, int>(max, i));
            }
        }
        
        splitCaps.Sort((a, b) => CompareCaps(a.Item1, b.Item1));
        
        prettyCaps = new List<string>();
        if (type == OpenSeeWebcamType.DirectShow)
            prettyCaps.Add("Default settings");
        foreach (var cap in splitCaps) {
            prettyCaps.Add(GetPrettyCapability(cap.Item1));
        }
        
        return prettyCaps;
    }
    
    public OpenSeeWebcamCapability GetCapabilityByPrettyIndex(int index) {
        int dshowOffset = 0;
        if (type == OpenSeeWebcamType.DirectShow) {
            if (index == 0) {
                OpenSeeWebcamCapability cap = new OpenSeeWebcamCapability();
                cap.id = -1;
                cap.format = OpenSeeWebcamFormat.Any;
                cap.minInterval = 10000000 / 60;
                return cap;
            }
            dshowOffset = 1;
        }
        
        if (splitCaps == null || index >= splitCaps.Count + dshowOffset || index < 0)
            throw new InvalidOperationException("Invalid capability index.");
        
        return caps[splitCaps[index - dshowOffset].Item2];
    }
}

[DefaultExecutionOrder(-50)]
public class OpenSeeWebcamInfo : MonoBehaviour {
    #region DllImport
    // DirectShow
    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "create_capture")]
	private static extern System.IntPtr create_capture_x64();

    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json_length")]
	private static extern int get_json_length_x64(System.IntPtr cap);

    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json")]
	private static extern void get_json_x64(System.IntPtr cap, [Out] StringBuilder namebuffer, int bufferlength);
    
    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "destroy_capture")]
	private static extern void destroy_capture_x64(System.IntPtr cap);

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "create_capture")]
	private static extern System.IntPtr create_capture_x86();

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json_length")]
	private static extern int get_json_length_x86(System.IntPtr cap);

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json")]
	private static extern void get_json_x86(System.IntPtr cap, [Out] StringBuilder namebuffer, int bufferlength);

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "destroy_capture")]
	private static extern void destroy_capture_x86(System.IntPtr cap);

    // Blackmagic
    [DllImport("libminibmcapture64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json_length")]
	private static extern int bm_get_json_length_x64();

    [DllImport("libminibmcapture64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json")]
	private static extern void bm_get_json_x64([Out] StringBuilder namebuffer, int bufferlength);

    [DllImport("libminibmcapture32", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json_length")]
	private static extern int bm_get_json_length_x86();

    [DllImport("libminibmcapture32", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_json")]
	private static extern void bm_get_json_x86([Out] StringBuilder namebuffer, int bufferlength);
    #endregion
    
    public List<OpenSeeWebcam> cameras;
    
    private bool initialized = false;

    static private string ListCameraDetails_x64() {
        System.IntPtr cap = create_capture_x64();
		int length = get_json_length_x64(cap);
        StringBuilder buffer = new StringBuilder(length);
        get_json_x64(cap, buffer, length);
        destroy_capture_x64(cap);
		return buffer.ToString();
    }

    static private string ListCameraDetails_x86() {
        System.IntPtr cap = create_capture_x86();
		int length = get_json_length_x86(cap);
        StringBuilder buffer = new StringBuilder(length);
        get_json_x86(cap, buffer, length);
        destroy_capture_x86(cap);
		return buffer.ToString();
    }

    static private string ListBlackMagicDetails_x64() {
		int length = bm_get_json_length_x64();
        StringBuilder buffer = new StringBuilder(length);
        bm_get_json_x64(buffer, length);
        //return "[{\"id\": 0,\"name\": \"DeckLink Mini Recorder\",\"bmId\": 2592763504,\"bmFlags\": \"1\",\"caps\": [{\"id\": 0,\"minCX\": 720,\"minCY\": 486,\"maxCX\": 720,\"maxCY\": 486,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 166833,\"maxInterval\": 166833,\"bmTimescale\": 60000,\"bmFrameduration\": 1001,\"bmModecode\": 1853125488,\"rating\": 1,\"format\": 100},{\"id\": 1,\"minCX\": 720,\"minCY\": 576,\"maxCX\": 720,\"maxCY\": 576,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 200000,\"maxInterval\": 200000,\"bmTimescale\": 50000,\"bmFrameduration\": 1000,\"bmModecode\": 1885432944,\"rating\": 1,\"format\": 100},{\"id\": 2,\"minCX\": 1920,\"minCY\": 1080,\"maxCX\": 1920,\"maxCY\": 1080,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 417083,\"maxInterval\": 417083,\"bmTimescale\": 24000,\"bmFrameduration\": 1001,\"bmModecode\": 842231923,\"rating\": 1,\"format\": 100},{\"id\": 3,\"minCX\": 1920,\"minCY\": 1080,\"maxCX\": 1920,\"maxCY\": 1080,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 416666,\"maxInterval\": 416666,\"bmTimescale\": 24000,\"bmFrameduration\": 1000,\"bmModecode\": 842297459,\"rating\": 1,\"format\": 100},{\"id\": 4,\"minCX\": 1920,\"minCY\": 1080,\"maxCX\": 1920,\"maxCY\": 1080,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 400000,\"maxInterval\": 400000,\"bmTimescale\": 25000,\"bmFrameduration\": 1000,\"bmModecode\": 1215312437,\"rating\": 1,\"format\": 100},{\"id\": 5,\"minCX\": 1920,\"minCY\": 1080,\"maxCX\": 1920,\"maxCY\": 1080,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 333666,\"maxInterval\": 333666,\"bmTimescale\": 30000,\"bmFrameduration\": 1001,\"bmModecode\": 1215312441,\"rating\": 1,\"format\": 100},{\"id\": 6,\"minCX\": 1920,\"minCY\": 1080,\"maxCX\": 1920,\"maxCY\": 1080,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 333333,\"maxInterval\": 333333,\"bmTimescale\": 30000,\"bmFrameduration\": 1000,\"bmModecode\": 1215312688,\"rating\": 1,\"format\": 100},{\"id\": 7,\"minCX\": 1280,\"minCY\": 720,\"maxCX\": 1280,\"maxCY\": 720,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 200000,\"maxInterval\": 200000,\"bmTimescale\": 50000,\"bmFrameduration\": 1000,\"bmModecode\": 1752184112,\"rating\": 1,\"format\": 100},{\"id\": 8,\"minCX\": 1280,\"minCY\": 720,\"maxCX\": 1280,\"maxCY\": 720,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 166833,\"maxInterval\": 166833,\"bmTimescale\": 60000,\"bmFrameduration\": 1001,\"bmModecode\": 1752184121,\"rating\": 1,\"format\": 100},{\"id\": 9,\"minCX\": 1280,\"minCY\": 720,\"maxCX\": 1280,\"maxCY\": 720,\"granularityCX\": 1,\"granularityCY\": 1,\"minInterval\": 166666,\"maxInterval\": 166666,\"bmTimescale\": 60000,\"bmFrameduration\": 1000,\"bmModecode\": 1752184368,\"rating\": 1,\"format\": 100}]}]";
		return buffer.ToString();
    }

    static private string ListBlackMagicDetails_x86() {
		int length = bm_get_json_length_x86();
        StringBuilder buffer = new StringBuilder(length);
        bm_get_json_x86(buffer, length);
		return buffer.ToString();
    }

    static public List<OpenSeeWebcam> ListCameraDetails() {
        string jsonData;
        string bmJsonData;
        if (Environment.Is64BitProcess) {
            jsonData = ListCameraDetails_x64();
            bmJsonData = ListBlackMagicDetails_x64();
        } else {
            jsonData = ListCameraDetails_x86();
            bmJsonData = ListBlackMagicDetails_x86();
        }
        List<OpenSeeWebcam> details = JsonConvert.DeserializeObject<List<OpenSeeWebcam>>(jsonData);
        foreach (var cam in details)
            cam.type = OpenSeeWebcamType.DirectShow;
        List<OpenSeeWebcam> bmDetails = JsonConvert.DeserializeObject<List<OpenSeeWebcam>>(bmJsonData);
        foreach (var cam in bmDetails)
            cam.type = OpenSeeWebcamType.Blackmagic;
        details.AddRange(bmDetails);
        return details;
    }
    
    void Start() {
        Initialize();
    }
    
    public void Initialize() {
        if (initialized)
            return;
        initialized = true;
        cameras = ListCameraDetails();
        foreach (var camera in cameras)
            camera.GetPrettyCapabilities();
    }
}
}