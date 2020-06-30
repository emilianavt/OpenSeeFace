using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using UnityEngine;
using OpenSee;
using VRM;

// This is a more comprehensive VRM avatar animator than the OpenSeeVRMExpression example.
// To use the lip sync functionality, place the OVRLipSync component somewhere in the scene.
// No other OVR related components are required.

public class OpenSeeVRMDriver : MonoBehaviour {
    #region DllImport
    [DllImport("user32.dll", SetLastError = true)]
    static extern ushort GetAsyncKeyState(int vKey);
    #endregion
    
    [Header("Settings")]
    [Tooltip("This is the OpenSeeExpression module used for expression prediction.")]
    public OpenSeeExpression openSeeExpression;
    [Tooltip("This is the OpenSeeIKTarget module used to move the avatar.")]
    public OpenSeeIKTarget openSeeIKTarget;
    [Tooltip("This is the target VRM avatar's blend shape proxy.")]
    public VRMBlendShapeProxy vrmBlendShapeProxy;
    //[Tooltip("When set to true, expression and audio data will be processed in FixedUpdate, otherwise it will be processed in Update.")]
    //public bool fixedUpdate = true;
    [Tooltip("When enabled, the avatar will blink automatically. Otherwise, blinks will be applied according to the face tracking's blink detection, allowing it to wink. Even for tracked eye blinks, the eye blink setting of the current expression is used.")]
    public bool autoBlink = true;
    [Tooltip("When automatic eye blinks are disabled, this sets the threshold for how closed the eyes have to be to be registered as fully closed. At 0 the eyes will never fully close.")]
    [Range(0, 1)]
    public float eyeClosedThreshold = 0.2f;
    [Tooltip("When automatic eye blinks are disabled, this sets the threshold for how open the eyes have to be to be registered as fully open. At 1 the eyes will never fully open.")]
    [Range(0, 1)]
    public float eyeOpenedThreshold = 0.55f;
    [Tooltip("This is the blink smoothing factor for camera based blink tracking, with 0 being no smoothing and 1 being a fixed blink state.")]
    [Range(0, 1)]
    public float blinkSmoothing = 0.75f;
    [Tooltip("When enabled, the blink state for both eyes is linked together.")]
    public bool linkBlinks = true;
    [Tooltip("When enabled, the avatar's eye will move according to the face tracker's gaze tracking.")]
    public bool gazeTracking = true;
    [Tooltip("This is the right eye bone. When either eye bone is not set, the VRM look direction blendshapes are used instead.")]
    public Transform rightEye;
    [Tooltip("This is the left eye bone. When either eye bone is not set, the VRM look direction blendshapes are used instead.")]
    public Transform leftEye;
    [Tooltip("This is the eyebrow tracking strength, with 0 meaning no eyebrow tracking and 1 meaning full strength.")]
    [Range(0, 1)]
    public float eyebrowStrength = 1.0f;
    [Tooltip("This is the eyebrow smoothing factor, with 0 being no smoothing and 1 being fixed eyebrows.")]
    public float eyebrowSmoothing = 0.65f;
    [Tooltip("This is the zero point of the eyebrow value. It can be used to offset the eyebrow tracking value.")]
    [Range(-1, 1)]
    public float eyebrowZero = 0.0f;
    [Tooltip("This is the sensitivity of the eyebrow tracking, with 0 meaning no eyebrow movement.")]
    [Range(0, 2)]
    public float eyebrowSensitivity = 1.0f;
    [Tooltip("This is the gaze smoothing factor, with 0 being no smoothing and 1 being a fixed gaze.")]
    [Range(0, 1)]
    public float gazeSmoothing = 0.6f;
    [Tooltip("This is the gaze stabilization factor, with 0 being no stabilization and 1 being a fixed gaze.")]
    [Range(0, 1)]
    public float gazeStabilizer = 0.1f;
    [Tooltip("For bone based gaze tracking, the conversion factor from [-1, 1] to degrees should be entered here. For blendshapes, sometimes gaze tracking does not follow the gaze strongly enough. With these factors, its effect can be strengthened, but usually it should be set to 1.")]
    public Vector2 gazeFactor = new Vector2(5f, 10f);
    [Tooltip("This factor is applied on top of the gazeFactor field.")]
    [Range(0, 5)]
    public float gazeStrength = 1f;
    [Tooltip("This this adds an offset with values in the range of -1 to 1 to the gaze coordinates.")]
    public Vector2 gazeCenter = Vector2.zero;
    [Tooltip("This component lets you customize the automatic eye blinking.")]
    public OpenSeeEyeBlink eyeBlinker = new OpenSeeEyeBlink();
    [Tooltip("When enabled, expressions will respond to global hotkeys.")]
    public bool hotkeys = true;
    [Tooltip("This component lets configure your VRM expressions.")]
    public OpenSeeVRMExpression[] expressions = new OpenSeeVRMExpression[]{
        new OpenSeeVRMExpression("neutral", BlendShapePreset.Neutral, true, 1f, 1f, 1f, true, true, 0x70, true, true, false, true, 50f, 120f),
        new OpenSeeVRMExpression("fun", BlendShapePreset.Fun, true, 1f, 1f, 1f, true, true, 0x71, true, true, false, true, 1f, 120f),
        new OpenSeeVRMExpression("joy", BlendShapePreset.Joy, true, 1f, 1f, 1f, true, true, 0x72, true, true, false, true, 1f, 120f),
        new OpenSeeVRMExpression("angry", BlendShapePreset.Angry, true, 1f, 1f, 0f, true, true, 0x73, true, true, false, true, 1f, 120f),
        new OpenSeeVRMExpression("sorrow", BlendShapePreset.Sorrow, true, 1f, 1f, 0f, true, true, 0x74, true, true, false, true, 1f, 120f),
        new OpenSeeVRMExpression("surprise", "Surprised", true, 1f, 1f, 0f, true, true, 0x75, true, true, false, true, 1f, 120f)
    };
    [Tooltip("The expression configuration is initialized on startup. If it is changed and needs to be reinitialized, this can be triggered by using this flag or calling InitExpressionMap. This flag is reset to false afterwards.")]
    public bool reloadExpressions = false;
    [Header("Lip sync settings")]
    [Tooltip("This allows you to enable and disable lip sync. When disabled, mouth tracking is used instead.")]
    public bool lipSync = true;
    [Tooltip("When enabled, camera based mouth tracking will be used when no viseme is detected.")]
    public bool hybridLipSync = false;
    [Tooltip("This is the mouth tracking smoothing factor, with 0 being no smoothing and 1 being a fixed mouth.")]
    [Range(0, 1)]
    public float mouthSmoothing = 0.6f;
    [Tooltip("This is the mouth tracking stabilization factor, with 0 being no stabilization and 1 being a fixed mouth.")]
    [Range(0, 1)]
    public float mouthStabilizer = 0.2f;
    [Tooltip("This is the mouth tracking stabilization factor used for wide mouth expressions, with 0 being no stabilization and 1 being a fixed mouth.")]
    [Range(0, 1)]
    public float mouthStabilizerWide = 0.3f;
    [Tooltip("When the detected audio volume is lower than this, the mouth will be considered closed for camera based mouth tracking.")]
    [Range(0, 1)]
    public float mouthSquelch = 0.01f;
    [Tooltip("With this, the squelch threshold above can be disabled.")]
    public bool mouthUseSquelch = false;
    [Tooltip("This allows you to select the OVRLipSync provider.")]
    public OVRLipSync.ContextProviders provider = OVRLipSync.ContextProviders.Enhanced;
    [Tooltip("When enabled, audio will be read from the given audio source, otherwise it will be read from the given mic.")]
    public bool useCanned = false;
    [Tooltip("This is the audio source for canned audio.")]
    public AudioSource audioSource = null;
    [Tooltip("This is the microphone audio will be captured from. A list can be retrieved from Microphone.devices.")]
    public string mic = null;
    [Tooltip("When enabled, the lip sync function will be initialized or reinitialized in the next Update or FixedUpdate call, according to the IK target's fixedUpdate flag. This flag is reset to false afterwards. It is also possible to call InitializeLipSync instead.")]
    public bool initializeLipSync = false;
    [Tooltip("This sets the viseme detection smoothing in a range from 1 to 100, where 1 means no smoothing and values above 90 mean pretty much no visemes.")]
    [Range(1, 100)]
    public int smoothAmount = 50;
    [Tooltip("This is the viseme animation smoothing factor, with 0 being no smoothing and 1 being no animation.")]
    [Range(0, 1)]
    public float visemeSmoothing = 0.75f;
    [Tooltip("Often, the A, I, U, E, O expressions are too strong, so you can scale them down using this value.")]
    [Range(0, 1)]
    public float visemeFactor = 0.6f;
    [Tooltip("This is the gain applied to audio read from the microphone. If your input level is high, set this back to 1. If your recording is very quiet and the lip sync has trouble picking up your speech, you can try increasing it.")]
    public float gain = 2f;
    [Header("Lip sync information")]
    [Tooltip("This shows if the lip sync system is currently active.")]
    public bool active = false;
    [Tooltip("This shows the current audio volume.")]
    public float audioVolume = 0f;

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
    private OpenSee.OpenSee.OpenSeeData openSeeData = null;
    private OpenSeeVRMExpression lastExpression = null;

    private float turnLeftBoundaryAngle = -30f;
    private float turnRightBoundaryAngle = 20f;
    
    private double lastGaze = 0f;
    private float lastLookUpDown = 0f;
    private float lastLookLeftRight = 0f;
    private float currentLookUpDown = 0f;
    private float currentLookLeftRight = 0f;
    private int interpolationCount = 0;
    private int interpolationState = 0;
    
    private float[] lastVisemeValues;
    private double lastMouth = 0f;
    private float[] lastMouthStates;
    private float[] currentMouthStates;
    private int mouthInterpolationCount = 0;
    private int mouthInterpolationState = 0;
    
    private double startedSilVisemes = -10;
    private double silVisemeHybridThreshold = 0.3;
    private bool wasSilViseme = true;
    
    private double lastBlink = 0f;
    private float lastBlinkLeft = 0f;
    private float lastBlinkRight = 0f;
    private float currentBlinkLeft = 0f;
    private float currentBlinkRight = 0f;
    private int blinkInterpolationCount = 0;
    private int blinkInterpolationState = 0;
    
    private bool overridden = false;
    private OpenSeeVRMExpression toggledExpression = null;
    private HashSet<OpenSeeVRMExpression> continuedPress = new HashSet<OpenSeeVRMExpression>();
    private bool currentExpressionEnableBlinking;
    private bool currentExpressionEnableVisemes;
    private float currentExpressionEyebrowWeight;
    private float currentExpressionVisemeFactor;
    
    private VRMBlendShapeProxy lastAvatar = null;
    private int eyebrowIsMoving = 0;
    private double lastBrows = 0f;
    private SkinnedMeshRenderer faceMesh;
    private int faceType = -1;
    private int browUpIndex = -1;
    private int browDownIndex = -1;
    private int browAngryIndex = -1;
    private int browSorrowIndex = -1;
    private float lastBrowUpDown = 0f;
    private float[] lastBrowStates;
    private float[] currentBrowStates;
    private int browInterpolationCount = 0;
    private int browInterpolationState = 0;
    private BlendShapeKey[] browClips;

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
        lastVisemeValues = new float[5] {0f, 0f, 0f, 0f, 0f};
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
        if (currentExpression != null)
            expressionFactor = currentExpressionVisemeFactor;
        if (currentExpression != null && !currentExpressionEnableVisemes) {
            return;
        }
        float weight;
        OVRLipSync.Viseme current = GetActiveViseme(out weight);
        if (current == OVRLipSync.Viseme.sil && weight > 0.9999f) {
            if (wasSilViseme) {
                if (hybridLipSync && Time.time - startedSilVisemes > silVisemeHybridThreshold) {
                    ApplyMouthShape();
                    return;
                }
            } else {
                startedSilVisemes = Time.time;
            }
            wasSilViseme = true;
        } else {
            if (wasSilViseme)
                startedSilVisemes = Time.time;
            wasSilViseme = false;
            if (hybridLipSync && Time.time - startedSilVisemes < silVisemeHybridThreshold)
                ApplyMouthShape(true);
        }
        weight = Mathf.Clamp(weight * 1.5f, 0f, 1f);
        float[] values = catsData[current];
        for (int i = 0; i < 5; i++) {
            lastVisemeValues[i] = values[i] * weight * (1f - visemeSmoothing) + lastVisemeValues[i] * visemeSmoothing;
            if (lastVisemeValues[i] < 0f) {
                lastVisemeValues[i] = 0f;
            }
            if (lastVisemeValues[i] > 1f) {
                lastVisemeValues[i] = 1f;
            }
            float result = lastVisemeValues[i] * visemeFactor * expressionFactor;
            if (result > 0f)
                vrmBlendShapeProxy.AccumulateValue(visemePresetMap[i], result);
        }
    }
    
    void ApplyMouthShape(bool fadeOnly) {
        if (vrmBlendShapeProxy == null)
            return;
        if (visemePresetMap == null)
            InitCatsData();
        if (openSeeData == null || openSeeData.features == null)
            return;
        
        if (lastMouth < openSeeData.time) {
            lastMouth = openSeeData.time;
            float open = openSeeData.features.MouthOpen;
            float wide = openSeeData.features.MouthWide;
            float[] mouthStates = new float[]{0f, 0f, 0f, 0f, 0f};
            float stabilizer = mouthStabilizer;
            float stabilizerWide = mouthStabilizerWide;
            
            if (openSeeData.rawEuler.y < turnLeftBoundaryAngle || openSeeData.rawEuler.y > turnRightBoundaryAngle) {
                stabilizer *= 2.5f;
                stabilizerWide *= 2.5f;
            }

            do {
                if (fadeOnly)
                    break;
                if (mouthUseSquelch && audioVolume < mouthSquelch)
                    break;
                if (open < stabilizer && Mathf.Abs(wide) < stabilizer)
                    break;
                if (wide > stabilizer && open < stabilizerWide)
                    break;
                if (open > 0.5f) {
                    // O
                    mouthStates[4] = open;
                } else if (open >= 0f) {
                    // A
                    mouthStates[0] = open * 1f;
                }
                if (wide >= 0f && open > stabilizer * 0.5f) {
                    if (wide > 0.5f) {
                        // I
                        mouthStates[1] = wide;
                    } else {
                        // E
                        mouthStates[3] = wide * 1f;
                    }
                } else if (wide < Mathf.Clamp(stabilizerWide * 1.5f, 0f, 0.8f) && open > stabilizer) {
                    // U
                    mouthStates[2] = -wide;
                }
                float total = 0f;
                float max = 0f;
                for (int i = 0; i < 5; i++) {
                    total += mouthStates[i];
                    if (mouthStates[i] > max)
                        max = mouthStates[i];
                }
                max = Mathf.Clamp(max * 3f, 0f, 1f);
                if (total < 0.0001f)
                    break;
                for (int i = 0; i < 5; i++) {
                    mouthStates[i] = max * (mouthStates[i] / total);
                }
            } while (false);
            
            if (mouthInterpolationState == 2) {
                for (int i = 0; i < 5; i++) {
                    mouthStates[i] = Mathf.Lerp(currentMouthStates[i], mouthStates[i], 1f - mouthSmoothing);
                }
            }
            
            lastMouthStates = currentMouthStates;
            currentMouthStates = mouthStates;
            if (mouthInterpolationState < 2)
                mouthInterpolationState++;
            mouthInterpolationCount = 0;
        }
        
        if (mouthInterpolationState < 2)
            return;
        
        float t = Mathf.Clamp((float)mouthInterpolationCount / openSeeIKTarget.averageInterpolations, 0f, 0.985f);
        mouthInterpolationCount++;
        
        if (currentExpression != null && !currentExpressionEnableVisemes)
            return;
        
        float expressionFactor = 1f;
        if (currentExpression != null)
            expressionFactor = currentExpressionVisemeFactor;

        for (int i = 0; i < 5; i++) {
            float interpolated = Mathf.Lerp(lastMouthStates[i], currentMouthStates[i], t);
            float result = interpolated * expressionFactor * visemeFactor;
            if (result > 0f)
                vrmBlendShapeProxy.AccumulateValue(visemePresetMap[i], result);
        }
    }
    
    void ApplyMouthShape() {
        ApplyMouthShape(false);
    }
    
    void UpdateBrows() {
        if ((browClips == null && (faceMesh == null || faceType < 0)) || openSeeData == null) {
            return;
        }
        if (lastBrows < openSeeData.time) {
            lastBrows = openSeeData.time;
            float upDownStrength = (openSeeData.features.EyebrowUpDownLeft + openSeeData.features.EyebrowUpDownRight) / 2f;
            float stabilizer = 0.3f;
            if (eyebrowZero > 0) {
                if (upDownStrength > eyebrowZero)
                    upDownStrength = (upDownStrength - eyebrowZero) / (1f - eyebrowZero);
                else
                    upDownStrength = (upDownStrength - eyebrowZero) / (1f + eyebrowZero);
            } else if (eyebrowZero < 0) {
                if (upDownStrength < eyebrowZero)
                    upDownStrength = (upDownStrength - eyebrowZero) / (1f + eyebrowZero);
                else
                    upDownStrength = (upDownStrength - eyebrowZero) / (1f - eyebrowZero);
            }

            float magnitude = Mathf.Clamp(Mathf.Abs(upDownStrength - lastBrowUpDown) * eyebrowSensitivity, -1f, 1f);
            float factor = 1f;
            if (openSeeData.rawEuler.y < turnLeftBoundaryAngle || openSeeData.rawEuler.y > turnRightBoundaryAngle) {
                factor = 0.7f;
                stabilizer *= 2;
            }
            if (upDownStrength > 0 && (eyebrowIsMoving > 0 || magnitude > stabilizer)) {
                upDownStrength = factor * Mathf.Clamp(upDownStrength / 0.6f, 0f, 1f);
                if (magnitude > stabilizer)
                    eyebrowIsMoving = 5;
            } else if (upDownStrength < 0 && (eyebrowIsMoving > 0 || magnitude > stabilizer)) {
                upDownStrength = -factor * Mathf.Clamp(upDownStrength / -0.2f, 0f, 1f);
                if (magnitude > stabilizer)
                    eyebrowIsMoving = 5;
            } else if (upDownStrength > stabilizer || upDownStrength < stabilizer) {
                upDownStrength = lastBrowUpDown;
            } else {
                upDownStrength = 0f;
            }
            if (eyebrowIsMoving > 0)
                eyebrowIsMoving--;
            upDownStrength = Mathf.Lerp(lastBrowUpDown, upDownStrength, 1f - eyebrowSmoothing);
            lastBrowUpDown = upDownStrength;
            
            if (browInterpolationState < 2)
                browInterpolationState++;
            browInterpolationCount = 0;
            
            for (int i = 0; i < 4; i++)
                lastBrowStates[i] = currentBrowStates[i];
            
            if (browClips != null) {
                currentBrowStates[0] = upDownStrength;
                currentBrowStates[1] = -upDownStrength;
            } else if (faceType == 0) {
                if (browUpIndex == -1 || browAngryIndex == -1 || browSorrowIndex == -1)
                    return;
                if (upDownStrength > 0) {
                    currentBrowStates[0] = upDownStrength * 100f;
                    currentBrowStates[2] = 0f;
                    currentBrowStates[3] = 0f;
                } else if (upDownStrength < 0) {
                    currentBrowStates[0] = 0f;
                    currentBrowStates[2] = -upDownStrength * 25f;
                    currentBrowStates[3] = -upDownStrength * 25f;
                } else {
                    currentBrowStates[0] = 0f;
                    currentBrowStates[2] = 0f;
                    currentBrowStates[3] = 0f;
                }
            } else if (faceType == 1) {
                if (browUpIndex == -1 || browDownIndex == -1 || browAngryIndex == -1 || browSorrowIndex == -1)
                    return;
                if (upDownStrength > 0) {
                    currentBrowStates[0] = upDownStrength * 100f;
                    currentBrowStates[1] = 0f;
                    currentBrowStates[2] = 0f;
                    currentBrowStates[3] = 0f;
                } else if (upDownStrength < 0) {
                    currentBrowStates[0] = 0f;
                    currentBrowStates[1] = -upDownStrength * 70f;
                    currentBrowStates[2] = 0f;
                    currentBrowStates[3] = 0f;
                } else {
                    currentBrowStates[0] = 0f;
                    currentBrowStates[1] = 0f;
                    currentBrowStates[2] = 0f;
                    currentBrowStates[3] = 0f;
                }
            }
        }
        
        float t = Mathf.Clamp((float)browInterpolationCount / openSeeIKTarget.averageInterpolations, 0f, 0.985f);
        browInterpolationCount++;
        
        float strength = eyebrowStrength;
        if (currentExpression != null)
            strength *= currentExpressionEyebrowWeight;
        
        if (browUpIndex > -1 && strength > 0f)
            faceMesh.SetBlendShapeWeight(browUpIndex, strength * Mathf.Lerp(lastBrowStates[0], currentBrowStates[0], t));
        if (browDownIndex > -1 && strength > 0f)
            faceMesh.SetBlendShapeWeight(browDownIndex, strength * Mathf.Lerp(lastBrowStates[1], currentBrowStates[1], t));
        if (browAngryIndex > -1 && strength > 0f)
            faceMesh.SetBlendShapeWeight(browAngryIndex, strength * Mathf.Lerp(lastBrowStates[2], currentBrowStates[2], t));
        if (browSorrowIndex > -1 && strength > 0f)
            faceMesh.SetBlendShapeWeight(browSorrowIndex, strength * Mathf.Lerp(lastBrowStates[3], currentBrowStates[3], t));
        if (browClips != null && strength > 0f) {
            if (lastBrowUpDown > 0f)
                vrmBlendShapeProxy.AccumulateValue(browClips[0], strength * Mathf.Lerp(lastBrowStates[0], currentBrowStates[0], t));
            else if (lastBrowUpDown < 0f)
                vrmBlendShapeProxy.AccumulateValue(browClips[1], strength * Mathf.Lerp(lastBrowStates[1], currentBrowStates[1], t));
        }
    }
    
    void FindFaceMesh() {
        if (lastAvatar == vrmBlendShapeProxy)
            return;
        lastAvatar = vrmBlendShapeProxy;
        
        faceMesh = null;
        faceType = -1;
        browUpIndex = -1;
        browDownIndex = -1;
        browAngryIndex = -1;
        browSorrowIndex = -1;
        browClips = null;
        lastBrowStates = new float[] {0f, 0f, 0f, 0f};
        currentBrowStates = new float[] {0f, 0f, 0f, 0f};
        
        if (vrmBlendShapeProxy == null)
            return;
        
        // Check if the model has set up blendshape clips. Thanks do Deat for this idea!
        string foundClipUp = null;
        string foundClipDown = null;
        foreach (BlendShapeClip clip in vrmBlendShapeProxy.BlendShapeAvatar.Clips) {
            if (clip.Preset == BlendShapePreset.Unknown) {
                if (clip.BlendShapeName.ToUpper() == "BROWS UP")
                    foundClipUp = clip.BlendShapeName;
                if (clip.BlendShapeName.ToUpper() == "BROWS DOWN")
                    foundClipDown = clip.BlendShapeName;
            }
        }
        if (foundClipUp != null && foundClipDown != null) {
            browClips =  new BlendShapeKey[2] {
                new BlendShapeKey(foundClipUp),
                new BlendShapeKey(foundClipDown)
            };
            return;
        }

        // Otherwise try to find direct blendshapes for VRoid and Cecil Henshin models.
        Component[] meshes = vrmBlendShapeProxy.gameObject.GetComponentsInChildren<SkinnedMeshRenderer>();
        foreach (SkinnedMeshRenderer renderer in meshes) {
            Mesh mesh = renderer.sharedMesh;
            
            for (int i = 0; i < mesh.blendShapeCount; i++) {
                string bsName = mesh.GetBlendShapeName(i);
                if (bsName.Contains("BRW") && bsName.Contains("Surprised")) {
                    browUpIndex = i;
                    faceType = 0;
                } else if (bsName.Contains("BRW") && bsName.Contains("Angry")) {
                    browAngryIndex = i;
                    faceType = 0;
                } else if (bsName.Contains("BRW") && bsName.Contains("Sorrow")) {
                    browSorrowIndex = i;
                    faceType = 0;
                } else if (bsName == "Oko") {
                    browAngryIndex = i;
                    faceType = 1;
                } else if (bsName == "Yowa") {
                    browSorrowIndex = i;
                    faceType = 1;
                } else if (bsName == "Down") {
                    browDownIndex = i;
                    faceType = 1;
                } else if (bsName == "Up") {
                    browUpIndex = i;
                    faceType = 1;
                }
            }
            if (faceType > -1) {
                faceMesh = renderer;
                return;
            }
        }
    }

    [Serializable]
    public class OpenSeeVRMExpression {
        [Tooltip("This is the expression string from the OpenSeeExpression component that will trigger this expression")]
        public string trigger = "neutral";
        [Tooltip("This is the VRM blend shape that will get triggered by the expression.")]
        public BlendShapePreset blendShapePreset;
        [Tooltip("When this name is set, it will be used to create a custom blendshape key.")]
        public string customBlendShapeName = "";
        [Tooltip("When disabled, this expression is considered a base expression. Only one base expression can be active at a time. Otherwise it can be added on top of other expressions.")]
        public bool additive = false;
        [Tooltip("The weight determines the value the expression blend shape should be set to.")]
        [Range(0, 1)]
        public float weight = 1f;
        [Tooltip("Some expressions involving the mouth require visemes to be weakened, which can be set using this factor. Setting it to 0 is the same as turning enableVisemes off.")]
        [Range(0, 1)]
        public float visemeFactor = 1f;
        [Tooltip("Some expressions involving the eyebrows require the eyebrow weight to be changed. Setting this weight to 0 will disable eyebrow tracking for this expression. Setting it to 1 enables it fully.")]
        [Range(0, 1)]
        public float eyebrowWeight = 1f;
        [Tooltip("This is the transition time for changing expressions. Setting it to 0 makes the transition instant.")]
        [Range(0, 1000)]
        public float transitionTime = 0f;
        [Tooltip("If the expression is not compatible with visemes, it can be turned off here.")]
        public bool enableVisemes = true;
        [Tooltip("If the expression is not compatible with blinking, it can be turned off here.")]
        public bool enableBlinking = true;
        [Tooltip("This can be set to a virtual key value. If this key is pressed together with shift and control, the expression will trigger as an override.")]
        public int hotkey = -1;
        [Tooltip("If this is set, the shift key has to be pressed for the hotkey to trigger.")]
        public bool shiftKey = false;
        [Tooltip("If this is set, the ctrl key has to be pressed for the hotkey to trigger.")]
        public bool ctrlKey = false;
        [Tooltip("If this is set, the alt key has to be pressed for the hotkey to trigger.")]
        public bool altKey = false;
        [Tooltip("If this is set, the expression will toggle, otherwise it will only be active while this is pressed.")]
        public bool toggle = true;
        [Tooltip("This is the weight for this class's error when training the model.")]
        public float errorWeight = 1f;
        [HideInInspector]
        public BlendShapeKey blendShapeKey;
        [HideInInspector]
        public bool toggled = false;
        [HideInInspector]
        public bool triggered = false;
        [HideInInspector]
        public bool reached = true;
        [HideInInspector]
        public bool lastActive = false;
        
        private float stateChangedTime = -1f;
        private float stateChangedWeight = 0f;
        private float lastWeight = 0f;
        private float targetWeight = 0f;
        private float remainingWeight = 0f;
        private float currentTransitionTime = 0f;
        
        public float GetWeight(float baseTransitionTime) {
            bool active = triggered || toggled;
            if (reached) {
                if (active) {
                    lastWeight = 1f;
                    return 1f;
                } else {
                    lastWeight = 0f;
                    return 0f;
                }
            }
            if (active != lastActive) {
                stateChangedTime = Time.time;
                stateChangedWeight = lastWeight;
                currentTransitionTime = transitionTime;
                lastActive = active;
                if (active) {
                    targetWeight = 1f;
                    remainingWeight = 1f - lastWeight;
                } else {
                    targetWeight = 0f;
                    remainingWeight = lastWeight;
                }
            }
            if (baseTransitionTime < currentTransitionTime)
                currentTransitionTime = baseTransitionTime;
            lastWeight = Mathf.Lerp(stateChangedWeight, targetWeight, Mathf.Min((Time.time - stateChangedTime) / (currentTransitionTime * 0.001f * remainingWeight), 1f));
            if (lastWeight >= 0.9999f && active) {
                lastWeight = 1f;
                reached = true;
            }
            if (lastWeight <= 0.0001f && !active) {
                lastWeight = 0f;
                reached = true;
            }
            return lastWeight;
        }

        public OpenSeeVRMExpression(string trigger, BlendShapePreset preset, bool additive, float weight, float factor, float eyebrows, bool visemes, bool blinking, int hotkey, bool shiftKey, bool ctrlKey, bool altKey, bool toggle, float errorWeight, float transitionTime) {
            this.trigger = trigger;
            this.blendShapePreset = preset;
            this.additive = additive;
            this.weight = weight;
            this.visemeFactor = factor;
            this.eyebrowWeight = eyebrows;
            this.enableVisemes = visemes;
            this.enableBlinking = blinking;
            this.hotkey = hotkey;
            this.shiftKey = shiftKey;
            this.ctrlKey = ctrlKey;
            this.altKey = altKey;
            this.toggle = toggle;
            this.errorWeight = errorWeight;
            this.transitionTime = transitionTime;
        }

        public OpenSeeVRMExpression(string trigger, string name, bool additive, float weight, float factor, float eyebrows, bool visemes, bool blinking, int hotkey, bool shiftKey, bool ctrlKey, bool altKey, bool toggle, float errorWeight, float transitionTime) {
            this.trigger = trigger;
            this.customBlendShapeName = name;
            this.additive = additive;
            this.weight = weight;
            this.visemeFactor = factor;
            this.eyebrowWeight = eyebrows;
            this.enableVisemes = visemes;
            this.enableBlinking = blinking;
            this.hotkey = hotkey;
            this.shiftKey = shiftKey;
            this.ctrlKey = ctrlKey;
            this.altKey = altKey;
            this.toggle = toggle;
            this.errorWeight = errorWeight;
            this.transitionTime = transitionTime;
        }
    }
    
    public OpenSeeVRMExpression GetExpression(string trigger) {
        if (!expressionMap.ContainsKey(trigger))
            return null;
        return expressionMap[trigger];
    }

    public void InitExpressionMap() {
        toggledExpression = null;
        overridden = false;
        continuedPress = new HashSet<OpenSeeVRMExpression>();
        expressionMap = new Dictionary<string, OpenSeeVRMExpression>();
        openSeeExpression.weightMap = new Dictionary<string, float>();
        foreach (var expression in expressions) {
            openSeeExpression.weightMap.Add(expression.trigger, expression.errorWeight);
            if (expression.customBlendShapeName != "")
                expression.blendShapeKey = new BlendShapeKey(expression.customBlendShapeName);
            else
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
        if (openSeeExpression == null) {
            currentExpression = null;
            return;
        }
        
        bool trigger = false;
        currentExpression = null;
        if (hotkeys) {
            bool shiftKey = (GetAsyncKeyState(0x10) & 0x8000U) != 0;
            bool ctrlKey = (GetAsyncKeyState(0x11) & 0x8000U) != 0;
            bool altKey = (GetAsyncKeyState(0x12) & 0x8000U) != 0;
            foreach (var expression in expressionMap.Values) {
                if (expression == toggledExpression && expression.toggled)
                    expression.triggered = true;
                else
                    expression.triggered = false;
                
                if (expression.hotkey >= 0 && expression.hotkey < 256 && (GetAsyncKeyState(expression.hotkey) & 0x8000) != 0 && shiftKey == expression.shiftKey && ctrlKey == expression.ctrlKey && altKey == expression.altKey) {
                    if (!continuedPress.Contains(expression)) {
                        // Generally, turn on expressions when their hotkey is pressed
                        expression.triggered = true;
                        if (expression.toggle) {
                            // If can be toggled, toggle it and update the triggered value accordingly
                            expression.toggled = !expression.toggled;
                            expression.triggered = expression.toggled;
                            // If it is a base expression
                            if (!expression.additive) {
                                if (expression.toggled) {
                                    // Override expression detection
                                    overridden = true;
                                    // Mark it as the current base expression while unmarking any other
                                    toggledExpression = expression;
                                    foreach (var otherExpression in expressionMap.Values) {
                                        if (otherExpression.toggle && !otherExpression.additive && otherExpression != expression) {
                                            otherExpression.toggled = false;
                                            otherExpression.triggered = false;
                                        }
                                    }
                                } else {
                                    // Unmark it as the current base expression if it was untoggled
                                    if (toggledExpression == expression) {
                                        toggledExpression = null;
                                        overridden = false;
                                    }
                                }
                            }
                            continuedPress.Add(expression);
                        } else if (!expression.additive) {
                            // If it's a base expression being triggered on, override expression detection
                            trigger = true;
                            currentExpression = expression;
                        }
                    }
                } else {
                    continuedPress.Remove(expression);
                }
                if (expression.lastActive != expression.triggered)
                    expression.reached = false;
            }
        }

        if (overridden)
            currentExpression = toggledExpression;
        
        if (!overridden && !trigger) {
            if (openSeeExpression.expression == null || openSeeExpression.expression == "" || !expressionMap.ContainsKey(openSeeExpression.expression)) {
                currentExpression = null;
            } else {
                currentExpression = expressionMap[openSeeExpression.expression];
            }
        }
        
        if (currentExpression == null && expressionMap.ContainsKey("neutral")) {
            currentExpression = expressionMap["neutral"];
        }

        // Reset limits on expressions
        currentExpressionEnableBlinking = true;
        currentExpressionEnableVisemes = true;
        currentExpressionEyebrowWeight = 1f;
        currentExpressionVisemeFactor = 1f;

        float baseTransitionTime = 1000f;
        if (currentExpression != null && !currentExpression.reached && currentExpression.transitionTime < baseTransitionTime)
            baseTransitionTime = currentExpression.transitionTime;
        if (lastExpression != currentExpression && lastExpression != null && !lastExpression.reached && lastExpression.transitionTime < baseTransitionTime)
            baseTransitionTime = lastExpression.transitionTime;
        foreach (var expression in expressionMap.Values) {
            float weight = expression.GetWeight(baseTransitionTime);
            if (weight > 0f) {
                vrmBlendShapeProxy.AccumulateValue(expression.blendShapeKey, weight);
                
                // Accumulate limits on expressions
                currentExpressionEnableBlinking = currentExpressionEnableBlinking && expression.enableBlinking;
                currentExpressionEnableVisemes = currentExpressionEnableVisemes && expression.enableVisemes;
                currentExpressionEyebrowWeight = Mathf.Min(currentExpressionEyebrowWeight, expression.eyebrowWeight);
                currentExpressionVisemeFactor = Mathf.Min(currentExpressionVisemeFactor, expression.visemeFactor);
            }
        }
        lastExpression = currentExpression;
        
        /*if (currentExpression != lastExpression) {
            expressionChangeTime = Time.time;
            lastExpression = currentExpression;
        }*/

        /*float timeWeight = 1f;
        if (currentExpression.transitionTime > 0f)
            timeWeight = Mathf.Clamp((Time.time - expressionChangeTime) / (currentExpression.transitionTime * 0.001f), 0f, 1f);
        if (timeWeight >= 1f) {
            foreach (var expression in expressionMap.Values) {
                if (expression != currentExpression) {
                    expression.maxWeight = 0f;
                    expression.lastTimeWeightOld = 0f;
                    vrmBlendShapeProxy.AccumulateValue(expression.blendShapeKey, 0f);
                }
            }
        } else {
            foreach (var expression in expressionMap.Values) {
                if (expression != currentExpression) {
                    float timeWeightOld = 1f;
                    if (expression.transitionTime > 0f)
                        timeWeightOld = Mathf.Max(Mathf.Clamp((Time.time - expressionChangeTime) / (expression.transitionTime * 0.001f), 0f, 1f), timeWeight);
                    if (expression.lastTimeWeightOld >= timeWeightOld)
                        expression.lastTimeWeightOld = timeWeightOld;
                    expression.maxWeight = Mathf.Clamp(expression.maxWeight - (timeWeightOld - expression.lastTimeWeightOld), 0f, 1f);
                    expression.lastTimeWeightOld = timeWeightOld;
                    vrmBlendShapeProxy.AccumulateValue(expression.blendShapeKey, expression.weight * expression.maxWeight);
                }
            }
        }
        
        currentExpression.maxWeight = timeWeight;
        currentExpression.lastTimeWeightOld = 0f;
        
        if (currentExpression != null)
            vrmBlendShapeProxy.AccumulateValue(currentExpression.blendShapeKey, currentExpression.weight * currentExpression.maxWeight);*/
    }
    
    void GetLookParameters(ref float lookLeftRight, ref float lookUpDown, bool update, int gazePoint, int right, int left, int topRight, int topLeft, int bottomRight, int bottomLeft) {
        float borderRight = (openSeeData.points3D[topRight].x + openSeeData.points3D[bottomRight].x) / 2.0f;
        float borderLeft = (openSeeData.points3D[topLeft].x + openSeeData.points3D[bottomLeft].x) / 2.0f;
        float horizontalCenter = (borderRight + borderLeft) / 2.0f;
        float horizontalRadius = (Mathf.Abs(horizontalCenter - borderRight) + Mathf.Abs(borderLeft - horizontalCenter)) / 2.0f;
        float borderTop = (openSeeData.points3D[topRight].y + openSeeData.points3D[topLeft].y) / 2.0f;
        float borderBottom = (openSeeData.points3D[bottomRight].y + openSeeData.points3D[bottomLeft].y) / 2.0f;
        float verticalCenter = (borderTop + borderBottom) / 2.0f;
        float verticalRadius = (Mathf.Abs(verticalCenter - borderTop) + Mathf.Abs(borderBottom - verticalCenter)) / 2.0f;
        float x = openSeeData.points3D[gazePoint].x;
        float y = openSeeData.points3D[gazePoint].y;
        float newLookLeftRight = Mathf.Clamp((x - horizontalCenter) / horizontalRadius + gazeCenter.x, -1f, 1f);
        float newLookUpDown = Mathf.Clamp((y - verticalCenter) / verticalRadius + gazeCenter.y, -1f, 1f);
        if (!update) {
            lookLeftRight = newLookLeftRight;
            lookUpDown = newLookUpDown;
        } else {
            lookLeftRight += newLookLeftRight;
            lookLeftRight /= 2.0f;
            lookUpDown += newLookUpDown;
            lookUpDown /= 2.0f;
        }
    }
    
    void UpdateGaze() {
        if (openSeeData == null || vrmBlendShapeProxy == null || openSeeIKTarget == null || !(openSeeIKTarget.averageInterpolations > 0f))
            return;
        
        float lookUpDown = currentLookUpDown;
        float lookLeftRight = currentLookLeftRight;

        if (lastGaze < openSeeData.time) {
            GetLookParameters(ref lookLeftRight, ref lookUpDown, false, 66, 36, 39, 37, 38, 41, 40);
            GetLookParameters(ref lookLeftRight, ref lookUpDown, true,  67, 42, 45, 43, 44, 47, 46);
            
            lookLeftRight = Mathf.Lerp(currentLookLeftRight, lookLeftRight, 1f - gazeSmoothing);
            lookUpDown = Mathf.Lerp(currentLookUpDown, lookUpDown, 1f - gazeSmoothing);

            if (Mathf.Abs(lookLeftRight - currentLookLeftRight) + Mathf.Abs(lookUpDown - currentLookUpDown) > gazeStabilizer) {
                lastLookLeftRight = currentLookLeftRight;
                lastLookUpDown = currentLookUpDown;
                currentLookLeftRight = lookLeftRight;
                currentLookUpDown = lookUpDown;
                lastGaze = openSeeData.time;
                if (interpolationState < 2)
                    interpolationState++;
                interpolationCount = 0;
            }
        }
        
        if (interpolationState < 2)
            return;
        
        float t = Mathf.Clamp((float)interpolationCount / openSeeIKTarget.averageInterpolations, 0f, 0.985f);
        lookLeftRight = Mathf.Lerp(lastLookLeftRight, currentLookLeftRight, t);
        lookUpDown = Mathf.Lerp(lastLookUpDown, currentLookUpDown, t);
        interpolationCount++;
        
        if (!openSeeIKTarget.mirrorMotion)
            lookLeftRight = -lookLeftRight;
        
        if (leftEye == null || rightEye == null) {
            vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookUp), 0f);
            vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookDown), 0f);
            vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookLeft), 0);
            vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookRight), 0f);
            if (gazeTracking) {
                if (lookUpDown > 0f) {
                    vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookUp), gazeFactor.x * lookUpDown);
                } else {
                    vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookDown), -gazeFactor.x * lookUpDown);
                }
                
                if (lookLeftRight > 0f) {
                    vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookLeft), gazeFactor.y * lookLeftRight);
                } else {
                    vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.LookRight), -gazeFactor.y * lookLeftRight);
                }
            }
        } else {
            //vrmLookAtHead.UpdateType = VRM.UpdateType.None;
            //vrmLookAtHead.RaiseYawPitchChanged(gazeFactor.y * lookLeftRight, gazeFactor.x * lookUpDown);
            rightEye.localRotation = Quaternion.identity;
            leftEye.localRotation = Quaternion.identity;
            if (gazeTracking) {
                Quaternion rotation = Quaternion.AngleAxis(-gazeFactor.x * gazeStrength * lookUpDown, Vector3.right) * Quaternion.AngleAxis(-gazeFactor.y * gazeStrength * lookLeftRight, Vector3.up);
                rightEye.localRotation = rotation;
                leftEye.localRotation = rotation;
            }
        }
    }

    void BlinkEyes() {
        if (vrmBlendShapeProxy == null || eyeBlinker == null)
            return;
        vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.Blink_R), 0f);
        vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.Blink_L), 0f);
        if (autoBlink) {
            if (currentExpressionEnableBlinking) {
                float blink = eyeBlinker.Blink();
                vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.Blink_R), blink);
                vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.Blink_L), blink);
            }
        } else if (openSeeExpression != null && openSeeExpression.openSee != null) {
            if (!currentExpressionEnableBlinking)
                return;

            float right = 1f;
            float left = 1f;
            
            if (openSeeData != null && lastBlink < openSeeData.time) {
                lastBlink = openSeeData.time;
                float openThreshold = eyeOpenedThreshold;
                
                if (openSeeData.rawEuler.y < turnLeftBoundaryAngle || openSeeData.rawEuler.y > turnRightBoundaryAngle)
                    openThreshold = Mathf.Lerp(openThreshold, eyeClosedThreshold, 0.4f);
                
                if (openSeeData.rightEyeOpen > openThreshold)
                    right = 0f;
                else if (openSeeData.rightEyeOpen < eyeClosedThreshold)
                    right = 1f;
                else
                    right = 1f - (openSeeData.rightEyeOpen - eyeClosedThreshold) / (openThreshold - eyeClosedThreshold);

                if (openSeeData.leftEyeOpen > openThreshold)
                    left = 0f;
                else if (openSeeData.leftEyeOpen < eyeClosedThreshold)
                    left = 1f;
                else
                    left = 1f - (openSeeData.leftEyeOpen - eyeClosedThreshold) / (openThreshold - eyeClosedThreshold);
                
                if (openSeeData.rawEuler.y < turnLeftBoundaryAngle)
                    left = right;
                if (openSeeData.rawEuler.y > turnRightBoundaryAngle)
                    right = left;
                
                lastBlinkLeft = currentBlinkLeft;
                lastBlinkRight = currentBlinkRight;
                currentBlinkLeft = Mathf.Lerp(currentBlinkLeft, left, 1f - blinkSmoothing);
                currentBlinkRight = Mathf.Lerp(currentBlinkRight, right, 1f - blinkSmoothing);
                
                if (left == 0f)
                    currentBlinkLeft = 0f;
                if (right == 0f)
                    currentBlinkRight = 0f;
                if (left == 1f)
                    currentBlinkLeft = 1f;
                if (right == 1f)
                    currentBlinkRight = 1f;
                
                if (blinkInterpolationState < 2)
                    blinkInterpolationState++;
            }
            
            if (blinkInterpolationState < 2)
                return;
            
            float t = Mathf.Clamp((float)blinkInterpolationCount / openSeeIKTarget.averageInterpolations, 0f, 0.985f);
            blinkInterpolationCount++;
            
            left = Mathf.Lerp(lastBlinkLeft, currentBlinkLeft, t);
            right = Mathf.Lerp(lastBlinkRight, currentBlinkRight, t);
            
            if (openSeeIKTarget.mirrorMotion) {
                float tmp = left;
                left = right;
                right = tmp;
            }
            
            if (linkBlinks) {
                float v = Mathf.Max(left, right);
                left = v;
                right = v;
            }
            
            vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.Blink_R), right);
            vrmBlendShapeProxy.AccumulateValue(new BlendShapeKey(BlendShapePreset.Blink_L), left);
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
        [Tooltip("When set, blinks will be smooth instead of instantaneous.")]
        public bool smoothBlink = true;
        private float blinkEyeRate;
        private float blinkEndTime = 1.0f;
        private float previousBlinkEyeRate;
        private float blinkEyeTime;
        private bool triggered = false;
        private float blinkStart = 0f;
        private float blinkDur = 0f;
        
        private float SmoothBlink() {
            if (!smoothBlink)
                return 1f;
            float fourth = blinkDur / 3f;
            float closeMax = blinkStart + fourth;
            float openMin = blinkStart + 2f * fourth;
            if (Time.time < closeMax)
                return (Time.time - blinkStart) / fourth;
            if (Time.time > openMin)
                return 1f - Mathf.Min((Time.time - openMin) / fourth, 1f);
            return 1f;
        }

        public float Blink()
        {
            float result = 0f;
            if (triggered && Time.time > blinkEndTime) {
                triggered = false;
                blinkEndTime = 0.0f;
                if (UnityEngine.Random.Range(0f, 1f) < doubleProb) {
                    blinkEyeTime = Time.time + UnityEngine.Random.Range(0.075f, 0.125f);
                }
            } else if (triggered) {
                result = SmoothBlink();
            }
            if (!triggered && Time.time > blinkEyeTime) {
                previousBlinkEyeRate = blinkEyeRate;
                blinkEyeTime = Time.time + blinkEyeRate;
                triggered = true;
                blinkStart = Time.time;
                blinkDur = UnityEngine.Random.Range(rangeDurLower, rangeDurUpper);
                if (blinkEndTime < 1.0f) {
                    result = SmoothBlink();
                }
                blinkEndTime = blinkStart + blinkDur;
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
        
        audioVolume = 0f;
        float leftVolume = 0f;
        float rightVolume = 0f;
        bool channel = false;
        foreach (float v in buffer) {
            float sample = Mathf.Abs(v);
            audioVolume += sample;
            if (channels > 1) {
                if (channel)
                    rightVolume += sample;
                channel = !channel;
            }
        }
        if (channels > 1) {
            leftVolume = audioVolume - rightVolume;
            if (rightVolume < leftVolume / 3f) {
                audioVolume = 2f * leftVolume;
                for (int i = 0; i + 1 < buffer.Length; i += 2) {
                    buffer[i + 1] = buffer[i];
                }
            } else if (leftVolume < rightVolume / 3f) {
                audioVolume = 2f * rightVolume;
                for (int i = 0; i + 1 < buffer.Length; i += 2) {
                    buffer[i] = buffer[i + 1];
                }
            }
        }
        audioVolume /= buffer.Length;

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
            if (isCanned)
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
        if (!isCanned)
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
        if (openSeeExpression != null && openSeeExpression.openSee != null) {
            openSeeData = openSeeExpression.openSee.GetOpenSeeData(openSeeExpression.faceId);
            if (openSeeData != null && openSeeData.fit3DError > openSeeExpression.openSee.maxFit3DError)
                openSeeData = null;
        }
        if (initializeLipSync) {
            InitializeLipSync();
            initializeLipSync = false;
        }
        if (vrmBlendShapeProxy != null) {
            foreach (var pair in vrmBlendShapeProxy.GetValues()) {
                vrmBlendShapeProxy.ImmediatelySetValue(pair.Key, 0f);
            }
        }
        FindFaceMesh();
        UpdateExpression();
        UpdateGaze();
        BlinkEyes();
        ReadAudio();
        if (lipSync) {
            ApplyVisemes();
        } else {
            ApplyMouthShape();
        }
        if (vrmBlendShapeProxy != null && browClips == null)
            vrmBlendShapeProxy.Apply();
        UpdateBrows();
        if (vrmBlendShapeProxy != null && browClips != null)
            vrmBlendShapeProxy.Apply();
    }

    void FixedUpdate() {
        if (openSeeIKTarget.fixedUpdate) {
            RunUpdates();
        }
    }

    void Update() {
        if (inited && lastSmoothing != smoothAmount) {
            OVRLipSync.SendSignal(context, OVRLipSync.Signals.VisemeSmoothing, smoothAmount, 0);
            lastSmoothing = smoothAmount;
        }
        if (!openSeeIKTarget.fixedUpdate) {
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