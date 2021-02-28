using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;

namespace OpenSee {

public class OpenSeeLauncher : MonoBehaviour {
    #region DllImport
    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "create_capture")]
	private static extern System.IntPtr create_capture_x64();

    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_devices")]
	private static extern int get_devices_x64(System.IntPtr cap);

	[DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_device")]
	private static extern void get_device_x64(System.IntPtr cap, int deviceno, [Out] StringBuilder namebuffer, int bufferlength);

    [DllImport("dshowcapture_x64", CallingConvention = CallingConvention.Cdecl, EntryPoint = "destroy_capture")]
	private static extern void destroy_capture_x64(System.IntPtr cap);

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "create_capture")]
	private static extern System.IntPtr create_capture_x86();

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_devices")]
	private static extern int get_devices_x86(System.IntPtr cap);

	[DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_device")]
	private static extern void get_device_x86(System.IntPtr cap, int deviceno, [Out] StringBuilder namebuffer, int bufferlength);

    [DllImport("dshowcapture_x86", CallingConvention = CallingConvention.Cdecl, EntryPoint = "destroy_capture")]
	private static extern void destroy_capture_x86(System.IntPtr cap);
    #endregion
    
    [Header("Settings")]
    [Tooltip("If this is left empty, it will default to the OpenSee component on the same game object.")]
    public OpenSee openSeeTarget;
    [Tooltip("The numeric index of the camera as in the list returned by the ListCameras function.")]
    public int cameraIndex = -1;
    [Tooltip("The path to a video file that should be used as input for the tracker instead of a camera")]
    public string videoPath = "";
    [Tooltip("Should facetracker.exe be started automatically with the scene?")]
    public bool autoStart = false;
    [Tooltip("If this is enabled, available cameras are automatically printed to the debug console.")]
    public bool autoListCameras = false;
    [Tooltip("If this is enabled, no output from the tracker will be printed to the debug console. This cannot be changed while the tracker is running.")]
    public bool dontPrint = true;
    [Tooltip("If enabled, the next available port within 500 starting after the one given on the OpenSee component will be used.")]
    public bool dynamicPort = false;
    [Tooltip("The path to \"facetracker.exe\".")]
    public string exePath = "facetracker.exe";
    [Tooltip("The path, where the .onnx model files can be found")]
    public string modelPath = "models\\";
    [Tooltip("Additional options that should be passed to the face tracker.")]
    public List<string> commandlineOptions = new List<string>(new string[] { "--silent", "1", "--max-threads", "1" });
    [Tooltip("This string will be appended at the end of the tracker options without quoting.")]
    public string extraOptions = "";
    [Tooltip("If enabled, the tracker commandline option string will be logged with Debug.Log.")]
    public bool logCommandline = false;
    [Tooltip("IL2CPP doesn't support Proocess.Start. When this is enabled, OpenSeeLauncher will create the tracking process by calling the necessary API functions directly through the DLLs. In this case, reading the standard output from the process will not be supported and it will instead be send to the logfiles set in pinvokeStdOut and pinvokeStdErr in the persistent data directory. It will also retrieve the camera list directly through the libdshowcapture DLLs, so make sure it is part of your Unity project.")]
    public bool usePinvoke = false;
    [Tooltip("When this is enabled, even if usePinvoke is disabled, the camera list will be retrieved through libdshowcapture DLLs directly, which can be faster. Make sure the DLLs are in your Unity project.")]
    public bool usePinvokeListCameras = false;
    [Tooltip("This is the standard output log file's name in the persistent data directory.")]
    public string pinvokeStdOut = "output.txt";
    [Tooltip("This is the standard error log file's name in the persistent data directory.")]
    public string pinvokeStdErr = "error.txt";
    [Header("Runtime information")]
    [Tooltip("This field shows if the tracker is currently alive.")]
    public bool trackerAlive = false;
    
    private string ip;
    private Process trackerProcess = null;
    private StringBuilder trackerSB = null;
    private bool dontPrintNow = false;
    private System.IntPtr processHandle = System.IntPtr.Zero;
    private System.IntPtr processStdOut = System.IntPtr.Zero;
    private System.IntPtr processStdErr = System.IntPtr.Zero;
    private Job job = null;
    
    public void UnsetOption(string name, bool hasArgument) {
        int count = 1;
        if (hasArgument)
            count++;
        int found = -1;
        int i = 0;
        foreach (string arg in commandlineOptions) {
            if (arg == name) {
                found = i;
                break;
            }
            i++;
        }
        if (found > -1)
            commandlineOptions.RemoveRange(found, count);
    }
    
    public void SetOption(string name, string argument) {
        UnsetOption(name, argument != null);
        commandlineOptions.Add(name);
        if (argument != null)
            commandlineOptions.Add(argument);
    }

    private bool CheckSetup(bool requireTarget) {
        if (openSeeTarget == null)
            openSeeTarget = GetComponent<OpenSee>();
        if (openSeeTarget == null) {
            UnityEngine.Debug.LogError("No openSeeTarget is set.");
            return false;
        }
        IPAddress address;
        if (!File.Exists(exePath)) {
            UnityEngine.Debug.LogError("Facetracker executable cannot be found.");
            return false;
        }
        /*if (!Directory.Exists(modelPath)) {
            UnityEngine.Debug.LogError("Model directory cannot be found.");
            return false;
        }*/
        if (requireTarget) {
            ip = openSeeTarget.listenAddress;
            if (!IPAddress.TryParse(ip, out address)) {
                UnityEngine.Debug.LogError("No valid IP address was given in the OpenSee component.");
                return false;
            }
            if (openSeeTarget.listenPort < 1 || openSeeTarget.listenPort > 65535) {
                UnityEngine.Debug.LogError("No valid port was given in the OpenSee component.");
                return false;
            }
            if (cameraIndex >= 0 && videoPath != "") {
                UnityEngine.Debug.LogError("Entering both a camera index and a video filename is not valid.");
                return false;
            }
            if (cameraIndex < 0 && !File.Exists(videoPath)) {
                if (videoPath != "") {
                    UnityEngine.Debug.LogError("The given video file does not exist.");
                } else {
                    UnityEngine.Debug.LogError("No camera or video file was given.");
                }
                return false;
            }
        }
        if (job == null) {
            job = new Job();
        }
        return true;
    }
    
    private string[] ListCameras_x64() {
		List<string> cameras = new List<string>();
        System.IntPtr cap = create_capture_x64();
		int count = get_devices_x64(cap);
		for (int i = 0; i < count; i++) {
			StringBuilder namebuffer = new StringBuilder(2048);
			get_device_x64(cap, i, namebuffer, 2048);
			cameras.Add(namebuffer.ToString());
		}
        destroy_capture_x64(cap);
		return cameras.ToArray();
    }
    
    private string[] ListCameras_x86() {
		List<string> cameras = new List<string>();
        System.IntPtr cap = create_capture_x86();
		int count = get_devices_x86(cap);
		for (int i = 0; i < count; i++) {
			StringBuilder namebuffer = new StringBuilder(2048);
			get_device_x86(cap, i, namebuffer, 2048);
			cameras.Add(namebuffer.ToString());
		}
        destroy_capture_x86(cap);
		return cameras.ToArray();
    }
    
    public string[] ListCameras() {
        if (usePinvoke || usePinvokeListCameras) {
            if (Environment.Is64BitProcess)
                return ListCameras_x64();
            else
                return ListCameras_x86();
        }
        
        if (!CheckSetup(false))
            return null;

        StringBuilder stringBuilder;
        ProcessStartInfo processStartInfo;
        Process process;

        stringBuilder = new StringBuilder();

        processStartInfo = new ProcessStartInfo();
        processStartInfo.CreateNoWindow = true;
        processStartInfo.RedirectStandardOutput = true;
        processStartInfo.RedirectStandardInput = true;
        processStartInfo.RedirectStandardError = true;
        processStartInfo.UseShellExecute = false;
        processStartInfo.FileName = exePath;
        processStartInfo.Arguments = "--list-cameras 2";
        
        process = new Process();
        process.StartInfo = processStartInfo;
        process.EnableRaisingEvents = true;
        process.OutputDataReceived += new DataReceivedEventHandler(delegate(object sender, DataReceivedEventArgs e) {
            stringBuilder.Append(e.Data);
            stringBuilder.Append("\n");
        });
        process.Start();
        job.AddProcess(process.Handle);
        process.BeginOutputReadLine();
        process.WaitForExit();
        process.CancelOutputRead();
        
        string cameraInfo = stringBuilder.ToString();
        string[] cameras = cameraInfo.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);
        return cameras;
    }
    
    public float[] Benchmark(int threads) {
        if (!CheckSetup(false))
            return null;

        StringBuilder stringBuilder;
        ProcessStartInfo processStartInfo;
        Process process;

        stringBuilder = new StringBuilder();

        processStartInfo = new ProcessStartInfo();
        processStartInfo.CreateNoWindow = true;
        processStartInfo.RedirectStandardOutput = true;
        processStartInfo.RedirectStandardInput = true;
        processStartInfo.RedirectStandardError = true;
        processStartInfo.UseShellExecute = false;
        processStartInfo.FileName = exePath;
        processStartInfo.Arguments = "--benchmark 1 --priority 4 --max-threads " + threads.ToString();
        
        process = new Process();
        process.StartInfo = processStartInfo;
        process.EnableRaisingEvents = true;
        process.OutputDataReceived += new DataReceivedEventHandler(delegate(object sender, DataReceivedEventArgs e) {
            stringBuilder.Append(e.Data);
            stringBuilder.Append("\n");
        });
        process.Start();
        job.AddProcess(process.Handle);
        process.BeginOutputReadLine();
        process.WaitForExit();
        process.CancelOutputRead();
        
        string[] lines = stringBuilder.ToString().Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);
        float[] results = new float[lines.Length];
        
        for (int i = 0; i < lines.Length; i++) {
            results[i] = Single.Parse(lines[i]);
        }
        
        return results;
    }
    
    // From: http://csharptest.net/529/how-to-correctly-escape-command-line-arguments-in-c/index.html
    private static string EscapeArguments(params string[] args)
    {
        StringBuilder arguments = new StringBuilder();
        Regex invalidChar = new Regex("[\x00\x0a\x0d]");//  these can not be escaped
        Regex needsQuotes = new Regex(@"\s|""");//          contains whitespace or two quote characters
        Regex escapeQuote = new Regex(@"(\\*)(""|$)");//    one or more '\' followed with a quote or end of string
        for (int carg = 0; args != null && carg < args.Length; carg++)
        {
            if (args[carg] == null) { throw new ArgumentNullException("args[" + carg + "]"); }
            if (invalidChar.IsMatch(args[carg])) { throw new ArgumentOutOfRangeException("args[" + carg + "]"); }
            if (args[carg] == String.Empty) { arguments.Append("\"\""); }
            else if (!needsQuotes.IsMatch(args[carg])) { arguments.Append(args[carg]); }
            else
            {
                arguments.Append('"');
                arguments.Append(escapeQuote.Replace(args[carg], m => m.Groups[1].Value + m.Groups[1].Value + (m.Groups[2].Value == "\"" ? "\\\"" : "")));
                arguments.Append('"');
            }
            if (carg + 1 < args.Length)
                arguments.Append(' ');
        }
        return arguments.ToString();
    }
    
    public bool StartTracker() {
        if (!CheckSetup(true))
            return false;

        List<string> arguments = new List<string>();
        arguments.Add("--ip");
        arguments.Add(ip);
        
        if (dynamicPort && !openSeeTarget.listening) {
            System.Net.IPEndPoint[] inUse = System.Net.NetworkInformation.IPGlobalProperties.GetIPGlobalProperties().GetActiveUdpListeners();
            Array.Sort(inUse, delegate(System.Net.IPEndPoint a, System.Net.IPEndPoint b) { return a.Port.CompareTo(b.Port); });
            int port = openSeeTarget.listenPort + 1;
            for (int i = 0; i < inUse.Length; i++) {
                if (inUse[i].Port < port)
                    continue;
                if (inUse[i].Port == port) {
                    if (port <= openSeeTarget.listenPort + 500 && port <= IPEndPoint.MaxPort) {
                        port++;
                        continue;
                    } else {
                        return false;
                    }
                }
                break;
            }
            openSeeTarget.listenPort = port;
        }

        arguments.Add("--port");
        arguments.Add(openSeeTarget.listenPort.ToString());
        arguments.Add("--model-dir");
        arguments.Add(modelPath);
        
        arguments.Add("--capture");
        if (videoPath != "" && File.Exists(videoPath))
            arguments.Add(videoPath);
        else
            arguments.Add(cameraIndex.ToString());
        foreach (string argument in commandlineOptions) {
            arguments.Add(argument);
        }
        string argumentString = EscapeArguments(arguments.ToArray());
        if (extraOptions != "")
            argumentString = argumentString + " " + extraOptions;
        
        if (logCommandline)
            UnityEngine.Debug.Log("Starting tracker: " + argumentString);

        StopTracker();

        if (!usePinvoke) {
            ProcessStartInfo processStartInfo;
            processStartInfo = new ProcessStartInfo();
            processStartInfo.CreateNoWindow = true;
            processStartInfo.RedirectStandardOutput = true;
            processStartInfo.RedirectStandardInput = true;
            processStartInfo.RedirectStandardError = true;
            processStartInfo.UseShellExecute = false;
            processStartInfo.FileName = exePath;
            processStartInfo.Arguments = argumentString;
            
            trackerSB = new StringBuilder();
            trackerProcess = new Process();
            trackerProcess.StartInfo = processStartInfo;
            dontPrintNow = dontPrint;
            if (!dontPrintNow) {
                trackerProcess.EnableRaisingEvents = true;
                trackerProcess.OutputDataReceived += new DataReceivedEventHandler(delegate(object sender, DataReceivedEventArgs e) {
                    trackerSB.Append(e.Data);
                    trackerSB.Append("\n");
                });
                trackerProcess.ErrorDataReceived += new DataReceivedEventHandler(delegate(object sender, DataReceivedEventArgs e) {
                    trackerSB.Append(e.Data);
                    trackerSB.Append("\n");
                });
            }
            trackerProcess.Start();
            job.AddProcess(trackerProcess.Handle);
            if (!dontPrintNow) {
                trackerProcess.BeginOutputReadLine();
                trackerProcess.BeginErrorReadLine();
            }

            return !trackerProcess.HasExited;
        } else {
            string dir = Path.GetDirectoryName(exePath);
            string outputLog = Application.persistentDataPath + "/" + pinvokeStdOut;
            string errorLog = Application.persistentDataPath + "/" + pinvokeStdErr;
            OpenSeeProcessInterface.Start(exePath, "facetracker " + argumentString, dir, true, outputLog, errorLog, out processHandle, out processStdOut, out processStdErr);
            if (processHandle != System.IntPtr.Zero) {
                job.AddProcess(processHandle);
                return OpenSeeProcessInterface.Alive(processHandle);
            } else
                return false;
        }
    }
    
    public void StopTracker() {
        if (processHandle != System.IntPtr.Zero) {
            if (OpenSeeProcessInterface.Alive(processHandle))
                OpenSeeProcessInterface.TerminateProcess(processHandle, 0);
            OpenSeeProcessInterface.CloseHandle(processHandle);
            processHandle = System.IntPtr.Zero;
        }
        if (processStdOut != System.IntPtr.Zero) {
            OpenSeeProcessInterface.CloseHandle(processStdOut);
            processStdOut = System.IntPtr.Zero;
        }
        if (processStdErr != System.IntPtr.Zero) {
            OpenSeeProcessInterface.CloseHandle(processStdErr);
            processStdErr = System.IntPtr.Zero;
        }
        if (trackerProcess != null && !trackerProcess.HasExited) {
            trackerProcess.CloseMainWindow();
            trackerProcess.Close();
            if (!dontPrintNow) {
                try {
                    trackerProcess.CancelOutputRead();
                    trackerProcess.CancelErrorRead();
                } catch {}
            }
        }
        trackerProcess = null;
    }
    
    public void Start() {
        if (autoStart)
            StartTracker();
        if (autoListCameras) {
            string[] cameras = ListCameras();
            if (cameras == null || cameras.Length < 1) {
                UnityEngine.Debug.Log("No cameras found.");
                return;
            }
            UnityEngine.Debug.Log("Cameras:");
            for (int i = 0; i < cameras.Length; i++) {
                UnityEngine.Debug.Log(i + " = " + cameras[i]);
            }
        }
    }
    
    public void Update() {
        if (processHandle != System.IntPtr.Zero) {
            trackerAlive = OpenSeeProcessInterface.Alive(processHandle);
            return;
        } else {
            if (trackerProcess != null)
                trackerAlive = !trackerProcess.HasExited;
            else {
                trackerAlive = false;
                return;
            }
        }
        if (dontPrintNow || trackerSB == null)
            return;
        int len = trackerSB.Length;
        if (len > 0) {
            string output = trackerSB.ToString();
            UnityEngine.Debug.Log("Tracker: " + output);
            trackerSB.Clear();
        }
    }
    
    private void CleanJob() {
        if (job != null) {
            job.Dispose();
            job = null;
        }
    }
    
    public void OnDestroy() {
        StopTracker();
        CleanJob();
    }
    
    public void OnApplicationQuit() {
        StopTracker();
        CleanJob();
    }
}

}