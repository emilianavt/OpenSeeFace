using System;
using System.Collections.Generic;
using UnityEngine;
using OpenSee;
using VRM;

// This is a more comprehensive VRM avatar animator than the OpenSeeVRMExpression example.
// To use the lip sync functionality, place the OVRLipSync component somewhere in the scene.
// No other OVR related components are required.

public class OpenSeeVRMDriver : MonoBehaviour {
    [Header("Settings")]
    [Tooltip("This is the OpenSeeExpression module used for expression prediction.")]
    public OpenSeeExpression openSeeExpression;
    [Tooltip("This is the target VRM avatar's blend shape proxy.")]
    public VRMBlendShapeProxy vrmBlendShapeProxy;
    [Tooltip("When set to true, expression and audio data will be processed in FixedUpdate, otherwise it will be processed in Update.")]
    public bool fixedUpdate = true;
    [Tooltip("When enabled, the avatar will blink automatically. Otherwise, blinks will be applied according to the face tracking's blink detection, allowing it to wink. Even for tracked eye blinks, the eye blink setting of the current expression is used.")]
    public bool autoBlink = true;
    [Tooltip("This component lets you customize the automatic eye blinking.")]
    public OpenSeeEyeBlink eyeBlinker = new OpenSeeEyeBlink();
    [Tooltip("This component lets configure your VRM expressions.")]
    public OpenSeeVRMExpression[] expressions = new OpenSeeVRMExpression[]{
        new OpenSeeVRMExpression("neutral", BlendShapePreset.Neutral, 1f, 1f, true, true),
        new OpenSeeVRMExpression("fun", BlendShapePreset.Fun, 1f, 1f, true, true),
        new OpenSeeVRMExpression("joy", BlendShapePreset.Joy, 1f, 1f, true, true),
        new OpenSeeVRMExpression("angry", BlendShapePreset.Angry, 1f, 1f, true, true),
        new OpenSeeVRMExpression("sorrow", BlendShapePreset.Sorrow, 1f, 1f, true, true)
    };
    [Tooltip("The expression configuration is initialized on startup. If it is changed and needs to be reinitialized, this can be triggered by using this flag or calling InitExpressionMap. This flag is reset to false afterwards.")]
    public bool reloadExpressions = false;
    [Header("Lip sync settings")]
    [Tooltip("This allows you to select the OVRLipSync provider.")]
    public OVRLipSync.ContextProviders provider = OVRLipSync.ContextProviders.Enhanced;
    [Tooltip("When enabled, audio will be read from the given audio source, otherwise it will be read from the given mic.")]
    public bool useCanned = false;
    [Tooltip("This is the audio source for canned audio.")]
    public AudioSource audioSource = null;
    [Tooltip("This is the microphone audio will be captured from. A list can be retrieved from Microphone.devices.")]
    public string mic = null;
    [Tooltip("When enabled, the lip sync function will be initialized or reinitialized in the next Update or FixedUpdate call, according to the fixedUpdate flag. This flag is reset to false afterwards. It is also possible to call InitializeLipSync instead.")]
    public bool initializeLipSync = false;
    [Range(1, 100)]
    [Tooltip("This sets the viseme smoothing in a range from 1 to 100, where 1 means no smoothing and values above 90 mean pretty much no visemes.")]
    public int smoothAmount = 50;
    [Tooltip("Often, the A, I, U, E, O expressions are too strong, so you can scale them down using this value.")]
    [Range(0, 1)]
    public float visemeFactor = 0.6f;
    [Tooltip("This is the gain applied to audio read from the microphone. If your input level is high, set this back to 1. If your recording is very quiet and the lip sync has trouble picking up your speech, you can try increasing it.")]
    public float gain = 2f;
    [Header("Lip sync information")]
    [Tooltip("This shows if the lip sync system is currently active.")]
    public bool active = false;

    private OVRLipSync.Frame visemeData = new OVRLipSync.Frame();
    private float volume;
    static private bool inited = false;
    private int lastSmoothing = 50;
    private uint context = 0;
    private float[] partialAudio;
    private int partialPos;
    private bool isCanned = false;
    private string lastMic;
    private int freq;
    private int lastPos = 0;
    private int channels = 0;
    private AudioClip clip = null;
    private BlendShapeKey[] visemePresetMap;
    private Dictionary<string, OpenSeeVRMExpression> expressionMap;
    private OpenSeeVRMExpression currentExpression = null;

    private Dictionary<OVRLipSync.Viseme, float[]> catsData = null;
    void InitCatsData() {
        catsData = new Dictionary<OVRLipSync.Viseme, float[]>();
        // This is similar to what the Blender CATS plugin does, but with A, I, U, E, O
        catsData.Add(OVRLipSync.Viseme.sil, new float[]{0f, 0f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.aa, new float[]{0.9998f, 0f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.CH, new float[]{0f, 0.9996f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.DD, new float[]{0.3f, 0.7f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.E, new float[]{0f, 0f, 0f, 0.9997f, 0f});
        catsData.Add(OVRLipSync.Viseme.FF, new float[]{0.2f, 0.4f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.ih, new float[]{0.5f, 0.2f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.kk, new float[]{0.7f, 0.4f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.nn, new float[]{0.2f, 0.7f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.oh, new float[]{0f, 0f, 0f, 0f, 0.9999f});
        catsData.Add(OVRLipSync.Viseme.ou, new float[]{0f, 0f, 0.9995f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.PP, new float[]{0.4f, 0f, 0f, 0f, 0.4f});
        catsData.Add(OVRLipSync.Viseme.RR, new float[]{0f, 0.5f, 0f, 0f, 0.3f});
        catsData.Add(OVRLipSync.Viseme.SS, new float[]{0f, 0.8f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.TH, new float[]{0.4f, 0f, 0f, 0f, 0.15f});
        visemePresetMap = new BlendShapeKey[5] {
            new BlendShapeKey(BlendShapePreset.A),
            new BlendShapeKey(BlendShapePreset.I),
            new BlendShapeKey(BlendShapePreset.U),
            new BlendShapeKey(BlendShapePreset.E),
            new BlendShapeKey(BlendShapePreset.O)
        };
    }

    OVRLipSync.Viseme GetActiveViseme(out float bestValue) {
        var visemes = Enum.GetValues(typeof(OVRLipSync.Viseme));
        bestValue = -1f;
        OVRLipSync.Viseme bestViseme = OVRLipSync.Viseme.sil;
        foreach (var viseme in visemes) {
            if (visemeData.Visemes[(int)viseme] > bestValue) {
                bestViseme = (OVRLipSync.Viseme)viseme;
                bestValue = visemeData.Visemes[(int)viseme];
            }
        }
        return bestViseme;
    }

    void ApplyVisemes() {
        if (vrmBlendShapeProxy == null || catsData == null)
            return;
        float expressionFactor = 1f;
        if (currentExpression != null && !currentExpression.enableVisemes) {
            for (int i = 0; i < 5; i++) {
                vrmBlendShapeProxy.ImmediatelySetValue(visemePresetMap[i], 0f);
            }
            expressionFactor = currentExpression.visemeFactor;
            return;
        }
        float weight;
        OVRLipSync.Viseme current = GetActiveViseme(out weight);
        weight = Mathf.Clamp(weight * 1.5f, 0f, 1f);
        float[] values = catsData[current];
        for (int i = 0; i < 5; i++) {
            vrmBlendShapeProxy.ImmediatelySetValue(visemePresetMap[i], values[i] * visemeFactor * expressionFactor * weight);
        }
    }

    [Serializable]
    public class OpenSeeVRMExpression {
        [Tooltip("This is the expression string from the OpenSeeExpression component that will trigger this expression")]
        public string trigger = "neutral";
        [Tooltip("This is the VRM blend shape that will get triggered by the expression.")]
        public BlendShapePreset blendShapePreset;
        [Tooltip("The weight determines the value the expression blend shape should be set to.")]
        [Range(0, 1)]
        public float weight = 1f;
        [Tooltip("Some expressions involving the mouth require visemes to be weakened, which can be set using this factor. Setting it to 0 is the same as turning enableVisemes off.")]
        [Range(0, 1)]
        public float visemeFactor = 1f;
        [Tooltip("If the expression is not compatible with visemes, it can be turned off here.")]
        public bool enableVisemes = true;
        [Tooltip("If the expression is not compatible with blinking, it can be turned off here.")]
        public bool enableBlinking = true;
        [HideInInspector]
        public BlendShapeKey blendShapeKey;

        public OpenSeeVRMExpression(string trigger, BlendShapePreset preset, float weight, float factor, bool visemes, bool blinking) {
            this.trigger = trigger;
            this.blendShapePreset = preset;
            this.weight = weight;
            this.visemeFactor = factor;
            this.enableVisemes = visemes;
            this.enableBlinking = blinking;
        }
    }

    public void InitExpressionMap() {
        expressionMap = new Dictionary<string, OpenSeeVRMExpression>();
        foreach (var expression in expressions) {
            expression.blendShapeKey = new BlendShapeKey(expression.blendShapePreset);
            expressionMap.Add(expression.trigger, expression);
        }
    }

    void UpdateExpression() {
        if (reloadExpressions) {
            InitExpressionMap();
            reloadExpressions = false;
        }
        if (vrmBlendShapeProxy == null) {
            currentExpression = null;
            return;
        }
        foreach (var expression in expressionMap.Values) {
            vrmBlendShapeProxy.ImmediatelySetValue(expression.blendShapeKey, 0f);
        }
        if (openSeeExpression == null) {
            currentExpression = null;
            return;
        }
        if (openSeeExpression.expression == null || openSeeExpression.expression == "" || !expressionMap.ContainsKey(openSeeExpression.expression)) {
            currentExpression = null;
            return;
        }
        currentExpression = expressionMap[openSeeExpression.expression];
        vrmBlendShapeProxy.ImmediatelySetValue(currentExpression.blendShapeKey, currentExpression.weight);
    }

    void BlinkEyes() {
        if (vrmBlendShapeProxy == null || eyeBlinker == null)
            return;
        if (autoBlink) {
            if (eyeBlinker.Blink() && (currentExpression == null || currentExpression.enableBlinking)) {
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_R), 1f);
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_L), 1f);
            } else {
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_R), 0f);
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_L), 0f);
            }
        } else if (openSeeExpression != null && openSeeExpression.openSee != null) {
            OpenSee.OpenSee.OpenSeeData openSeeData = openSeeExpression.openSee.GetOpenSeeData(openSeeExpression.faceId);
            if (openSeeData == null || !currentExpression.enableBlinking) {
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_R), 0f);
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_L), 0f);
                return;
            }
            float right = 1f;
            if (openSeeData.rightEyeOpen > 0.5f)
                right = 0f;
            float left = 1f;
            if (openSeeData.leftEyeOpen > 0.5f)
                left = 0f;
            vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_R), right);
            vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_L), left);
        }
    }

    [Serializable]
    public class OpenSeeEyeBlink {
        [Tooltip("This is the minimum duration between eye blinks.")]
        public float rangeLower = 1.2f;
        [Tooltip("This is the maximum duration between eye blinks.")]
        public float rangeUpper = 5.1f;
        [Tooltip("This is the probability of blinking twice in a row.")]
        public float doubleProb = 0.05f;
        [Tooltip("This is the minimum duration of eye blinks.")]
        public float rangeDurLower = 0.1f;
        [Tooltip("This is the maximum duration of eye blinks.")]
        public float rangeDurUpper = 0.2f;
        private float blinkEyeRate;
        private float blinkEndTime = 1.0f;
        private float previousBlinkEyeRate;
        private float blinkEyeTime;
        private bool triggered = false;
        private bool result = false;

        public bool Blink()
        {
            if (triggered && Time.time > blinkEndTime) {
                result = false;
                triggered = false;
                blinkEndTime = 0.0f;
                if (UnityEngine.Random.Range(0f, 1f) < doubleProb) {
                    blinkEyeTime = Time.time + UnityEngine.Random.Range(0.075f, 0.125f);
                }
            } else if (triggered) {
                result = true;
            }
            if (!triggered && Time.time > blinkEyeTime) {
                previousBlinkEyeRate = blinkEyeRate;
                blinkEyeTime = Time.time + blinkEyeRate;
                if (blinkEndTime < 1.0f) {
                    result = true;
                }
                triggered = true;
                blinkEndTime = Time.time + UnityEngine.Random.Range(rangeDurLower, rangeDurUpper);
                while (previousBlinkEyeRate == blinkEyeRate) {
                     blinkEyeRate = UnityEngine.Random.Range(rangeLower, rangeUpper);
                }
            }
            return result;
        }
    }

    public void InitializeLipSync() {
        active = false;
        if (catsData == null)
            InitCatsData();
        if (!isCanned && clip != null)
            Microphone.End(lastMic);

        partialAudio = null;
        partialPos = 0;

        audioSource = GetComponent<AudioSource>();
        if (useCanned && audioSource != null && audioSource.clip != null) {
            isCanned = true;
            clip = audioSource.clip;
            channels = clip.channels;
            partialAudio = new float[1024 * channels];
            freq = audioSource.clip.frequency;
            if (!inited) {
                if (OVRLipSync.IsInitialized() == OVRLipSync.Result.Success) {
                    DestroyContext();
                    OVRLipSync.Shutdown();
                }
                OVRLipSync.Initialize(freq, 1024);
                CreateContext();
                OVRLipSync.SendSignal(context, OVRLipSync.Signals.VisemeSmoothing, smoothAmount, 0);
                inited = true;
            }
            active = true;
            return;
        }
        
        isCanned = false;

        int minFreq;
        int maxFreq = AudioSettings.outputSampleRate;
        freq = maxFreq;

        lastMic = mic;
        if (mic != null)
            Microphone.GetDeviceCaps(lastMic, out minFreq, out maxFreq);
        if (maxFreq > 0)
            freq = maxFreq;

        if (!inited) {
            if (OVRLipSync.IsInitialized() == OVRLipSync.Result.Success) {
                DestroyContext();
                OVRLipSync.Shutdown();
            }
            OVRLipSync.Initialize(freq, 1024);
            CreateContext();
            OVRLipSync.SendSignal(context, OVRLipSync.Signals.VisemeSmoothing, smoothAmount, 0);
            inited = true;
        }

        clip = Microphone.Start(lastMic, true, 1, freq);
        channels = clip.channels;
        partialAudio = new float[1024 * channels];
        lastPos = 0;
        active = true;
    }

    float[] ReadMic() {
        if (clip == null)
            return null;
        if (isCanned)
            return null;
        int pos = Microphone.GetPosition(lastMic);
        if (pos < 0 || pos == lastPos)
            return null;
        int len = pos - lastPos;
        if (lastPos > pos)
            len = pos + clip.samples - lastPos;
        if (clip.channels != channels) {
            if (clip.channels > 2) {
                Debug.Log("Audio with more than 3 channels is not supported.");
                Microphone.End(lastMic);
                clip = null;
                active = false;
                return null;
            }
            channels = clip.channels;
            partialAudio = new float[1024 * channels];
        }
        float[] buffer = new float[len * channels];
        clip.GetData(buffer, lastPos);
        lastPos = pos;
        return buffer;
    }

    void ProcessBuffer(float[] buffer) {
        if (buffer == null)
            return;
        int totalLen = partialPos + buffer.Length;
        int bufferPos = 0;
        if (totalLen >= 1024 * channels) {
            volume = 0f;
            while (totalLen >= 1024 * channels) {
                int sliceLen = 1024 - partialPos;
                Array.Copy(buffer, bufferPos, partialAudio, partialPos, sliceLen * channels);
                totalLen -= 1024 * channels;
                if (totalLen < 1024 * channels) {
                    for (int i = 0; i < partialAudio.Length; i++) {
                        partialAudio[i] = partialAudio[i] * gain;//Mathf.Clamp(partialAudio[i] * gain, 0f, 1f);
                        volume += Mathf.Abs(partialAudio[i]);
                    }
                    lock (this) {
                        if (context != 0) {
                            OVRLipSync.Frame frame = this.visemeData;
                            if (channels == 2)
                                OVRLipSync.ProcessFrameInterleaved(context, partialAudio, frame);
                            else
                                OVRLipSync.ProcessFrame(context, partialAudio, frame);
                        } else {
                            Debug.Log("OVRLipSync context is 0");
                        }
                    }
                }
                bufferPos += sliceLen;
                partialPos = 0;
            }
            volume /= (float)buffer.Length;
        }
        if (totalLen > 0) {
            Array.Copy(buffer, bufferPos, partialAudio, partialPos, buffer.Length - bufferPos);
            partialPos += (buffer.Length - bufferPos) / channels;
        }
    }

    void OnAudioFilterRead(float[] data, int channels) {
        if (isCanned) {
            if (channels > 3) {
                Debug.Log("Audio with more than 3 channels is not supported.");
                clip = null;
                isCanned = false;
                active = false;
                return;
            }
            float[] buffer = new float[data.Length];
            Array.Copy(data, buffer, data.Length);
            if (channels != this.channels) {
                this.channels = channels;
                partialAudio = new float[1024 * channels];
            }
            ProcessBuffer(buffer);
        }
    }

    void ReadAudio() {
        if (clip == null)
            return;
        if (OVRLipSync.IsInitialized() != OVRLipSync.Result.Success) {
            Debug.Log("OVRLipSync is not ready.");
            return;
        }
        float[] buffer = ReadMic();
        if (buffer == null)
            return;
        ProcessBuffer(buffer);
    }

    void CreateContext() {
        lock (this) {
            if (context == 0) {
                if (OVRLipSync.CreateContext(ref context, provider) != OVRLipSync.Result.Success) {
                    Debug.LogError("OVRLipSyncContextBase.Start ERROR: Could not create Phoneme context.");
                    return;
                }
            }
        }
    }

    void Start() {
        InitExpressionMap();
    }

    void RunUpdates() {
        if (initializeLipSync) {
            InitializeLipSync();
            initializeLipSync = false;
        }
        UpdateExpression();
        BlinkEyes();
        ReadAudio();
        ApplyVisemes();
    }

    void FixedUpdate() {
        if (fixedUpdate) {
            RunUpdates();
        }
    }

    void Update() {
        if (inited && lastSmoothing != smoothAmount) {
            OVRLipSync.SendSignal(context, OVRLipSync.Signals.VisemeSmoothing, smoothAmount, 0);
            lastSmoothing = smoothAmount;
        }
        if (!fixedUpdate) {
            RunUpdates();
        }
    }

    void DestroyContext() {
        active = false;
        lock (this) {
            if (context != 0) {
                if (OVRLipSync.DestroyContext(context) != OVRLipSync.Result.Success) {
                    Debug.LogError("OVRLipSyncContextBase.OnDestroy ERROR: Could not delete Phoneme context.");
                }
                context = 0;
            }
        }
    }

    void OnDestroy() {
        DestroyContext();
    }
}