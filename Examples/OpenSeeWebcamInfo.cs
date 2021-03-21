using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
    
    bool ge(bool a, bool b) {
        return a || (!a && !b);
    }

    private int CompareCaps(OpenSeeWebcamCapability a, OpenSeeWebcamCapability b) {
        int aRes = a.minCX * a.minCY;
        int bRes = b.minCX * b.minCY;
        bool aGoodFps = a.minInterval <= 670000;
        bool bGoodFps = a.minInterval <= 670000;
        bool aGoodRes = aRes >= 850000 && aRes <= 2200000;
        bool bGoodRes = bRes >= 850000 && bRes <= 2200000;
        if (a.rating < b.rating && ge(aGoodFps, bGoodFps) && ge(aGoodRes, bGoodRes))
            return -1;
        if (a.rating > b.rating && ge(bGoodFps, aGoodFps) && ge(bGoodRes, aGoodRes))
            return 1;
        if (aRes > bRes && ge(aGoodFps, bGoodFps) && ge(aGoodRes, bGoodRes))
            return -1;
        if (aRes < bRes && ge(bGoodFps, aGoodFps) && ge(bGoodRes, aGoodRes))
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
    
    [Serializable]
    public class OpenSeeWebcamList {
        public List<OpenSeeWebcam> list;
    }

    public List<OpenSeeWebcam> cameras;
    public bool dumpJson = false;
    
    private static bool dumpJsonStatic = false;
    
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
		return buffer.ToString();
    }

    static private string ListBlackMagicDetails_x86() {
		int length = bm_get_json_length_x86();
        StringBuilder buffer = new StringBuilder(length);
        bm_get_json_x86(buffer, length);
		return buffer.ToString();
    }

    static public List<OpenSeeWebcam> ListCameraDetails(bool includeBlackMagic) {
        string jsonData = null;
        string bmJsonData = null;
        if (Environment.Is64BitProcess) {
            jsonData = ListCameraDetails_x64();
            if (includeBlackMagic)
                bmJsonData = ListBlackMagicDetails_x64();
        } else {
            jsonData = ListCameraDetails_x86();
            if (includeBlackMagic)
                bmJsonData = ListBlackMagicDetails_x86();
        }
        if (dumpJsonStatic) {
            UnityEngine.Debug.Log("Camera JSON: " + jsonData);
            if (includeBlackMagic)
                UnityEngine.Debug.Log("Blackmagic JSON: " + bmJsonData);
        }
        List<OpenSeeWebcam> details = JsonUtility.FromJson<OpenSeeWebcamList>("{\"list\":" + jsonData + "}").list;
        foreach (var cam in details)
            cam.type = OpenSeeWebcamType.DirectShow;
        if (includeBlackMagic) {
            List<OpenSeeWebcam> bmDetails = JsonUtility.FromJson<OpenSeeWebcamList>("{\"list\":" + bmJsonData + "}").list;
            foreach (var cam in bmDetails)
                cam.type = OpenSeeWebcamType.Blackmagic;
            details.AddRange(bmDetails);
        }
        return details;
    }
    
    void Start() {
        Initialize(false);
    }
    
    public void Initialize(bool includeBlackMagic) {
        dumpJsonStatic = dumpJson;
        cameras = ListCameraDetails(includeBlackMagic);
        foreach (var camera in cameras)
            camera.GetPrettyCapabilities();
    }
}
}