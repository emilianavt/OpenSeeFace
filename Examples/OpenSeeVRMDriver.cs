#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
#define WINDOWS_BUILD
#endif

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using UnityEngine;
using OpenSee;
using VRM;

// This is a more comprehensive VRM avatar animator than the OpenSeeVRMExpression example.
// To use the lip sync functionality, place the OVRLipSync 1.28 component somewhere in the scene.
// No other OVR related components are required.

public class OpenSeeVRMDriver : MonoBehaviour {
    #if WINDOWS_BUILD
    #region DllImport
    [DllImport("user32.dll", SetLastError = true)]
    static extern ushort GetAsyncKeyState(int vKey);
    #endregion
    #else
    static ushort GetAsyncKeyState(int vKey) { return 0; }
    #endif
    
    [Header("Settings")]
    [Tooltip("This is the OpenSeeExpression module used for expression prediction.")]
    public OpenSeeExpression openSeeExpression;
    [Tooltip("This is the OpenSeeIKTarget module used to move the avatar.")]
    public OpenSeeIKTarget openSeeIKTarget;
    [Tooltip("This is the target VRM avatar's blend shape proxy.")]
    public VRMBlendShapeProxy vrmBlendShapeProxy;
    [Tooltip("When this is enabled, no blendshapes are set.")]
    public bool skipApply = false;
    [Tooltip("When this is enabled, no gaze tracking (bone or blendshape) is applied.")]
    public bool skipApplyEyes = false;
    [Tooltip("When this is enabled, no jaw bone animation is applied.")]
    public bool skipApplyJaw = false;
    [Tooltip("When this is enabled in addition to skipApply, expressions from hotkeys or expression detection are still applied.")]
    public bool stillApplyExpressions = false;
    [Tooltip("This factor is multiplied to all blendshape weights relating to camera tracking, as well as gaze and jaw tracking.")]
    public float blendShapeWeight = 1f;
    [Tooltip("This needs to be enabled when tracking with the fast 30 point model.")]
    public bool only30Points = false;
    [Tooltip("When enabled, if a model has perfect sync blendshapes, they will be used for more detailed tracking. (Experimental!)")]
    public bool useCameraPerfectSync = false;
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
    [Tooltip("With tracked eye blinking, the eyes will close when looking down. Enable this to compensate this by modifying thresholds.")]
    public bool lookDownCompensation = false;
    [Tooltip("This is an angle value that can be used to adjust the angle from which a face is considered to be looking downwards.")]
    [Range(-10, 20)]
    public float lookDownAdjustment = 0f;
    [Tooltip("When enabled, auto blinking will be enabled while looking down.")]
    public bool autoBlinkLookingDown = false;
    [Tooltip("When enabled, the blink state for both eyes is permanently linked together.")]
    public bool linkBlinks = true;
    [Tooltip("When enabled, it becomes possible to wink even with link blinks enabled.")]
    public bool allowWinking = false;
    [Tooltip("If allowWinking is enabled, if the eye blink values are within this distance, the blink is linked.")]
    public float smartWinkThreshold = 0.85f;
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
    [Tooltip("When enabled, viseme size will not depend on the OVR Lip Sync weight.")]
    public bool maximizeLipSync = false;
    [Tooltip("When enabled, mouth blendshapes are normalized if they add up to a value above 1.")]
    public bool visemeNormalization = true;
    [Tooltip("When enabled and the animator of the avatar has a float parameter called JawMovement, this parameter will be set to values from 0 to 1, corresponding to a closed to open mouth. This can be used for animating jaw bones.")]
    public bool animateJawBone = false;
    [Tooltip("This should be set to the included mecanim animation, which sets the jaw bone to closed on the first frame and to open on the second frame.")]
    public AnimationClip jawBoneAnimation;
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
    #if WINDOWS_BUILD
    [Tooltip("This allows you to select the OVRLipSync provider.")]
    public OVRLipSync.ContextProviders provider = OVRLipSync.ContextProviders.Enhanced;
    #endif
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

    [HideInInspector]
    public OpenSee.OpenSee.OpenSeeData openSeeData { get; private set; } = null;

    #if WINDOWS_BUILD
    private OVRLipSync.Frame visemeData = new OVRLipSync.Frame();
    #endif
    private float fakeTimeTime = 0f;
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
    private OpenSeeVRMExpression lastExpression = null;

    private bool haveFullVisemeSet = false;
    private BlendShapeKey[] fullVisemePresets;
    #if WINDOWS_BUILD
    private Dictionary<OVRLipSync.Viseme, int> ovrMap;
    #endif

    private float turnLeftBoundaryAngle = -30f;
    private float turnRightBoundaryAngle = 20f;
    private float turnDownBoundaryAngle = 205f;
    private float wasLookingDown = 0f;
    
    private double lastGaze = 0f;
    private float lastLookUpDown = 0f;
    private float lastLookLeftRight = 0f;
    private float currentLookUpDown = 0f;
    private float currentLookLeftRight = 0f;
    private TimeInterpolate gazeInterpolate;
    
    private float[] lastVisemeValues;
    private double lastMouth = 0f;
    private float[] lastMouthStates;
    private float[] currentMouthStates;
    private TimeInterpolate mouthInterpolate;
    
    private double startedSilVisemes = -10;
    private double silVisemeHybridThreshold = 0.3;
    private bool wasSilViseme = true;
    
    private double lastBlink = 0f;
    private float lastBlinkLeft = 0f;
    private float lastBlinkRight = 0f;
    private float currentBlinkLeft = 0f;
    private float currentBlinkRight = 0f;
    private TimeInterpolate blinkInterpolate;
    
    private bool overridden = false;
    private OpenSeeVRMExpression toggledExpression = null;
    private HashSet<OpenSeeVRMExpression> continuedPress = new HashSet<OpenSeeVRMExpression>();
    private bool currentExpressionEnableBlinking;
    private bool currentExpressionEnableVisemes;
    private float currentExpressionEyebrowWeight;
    private float currentExpressionVisemeFactor;
    
    private OpenSeeBlendShapeProxy proxy = null;
    private Animator animator;
    private bool haveJawParameter;
    private Transform jawBone;
    private Quaternion jawRotation = Quaternion.identity;
    private HumanBodyBones[] humanBodyBones;
    private Transform[] humanBodyBoneTransforms;
    private Quaternion[] humanBodyBoneRotations;
    
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
    private TimeInterpolate browInterpolate;
    private BlendShapeKey[] browClips;
    
    private bool trackMouth = false;
    private double lastPerfectSync = -1;
    
    private float lastAudioTime = -1f;

    private class TimeInterpolate {
        private float interpolateT = 0f;
        private float updateT = 0f;
        private double lastT = 0f;
        private double currentT = 0f;
        private float clamp = 1f;
        private int gotData = 0;
        
        public TimeInterpolate() {}
        public TimeInterpolate(float clamp) {
            this.clamp = clamp;
        }
        
        public void UpdateTime(double nowT) {
            updateT = Time.time;
            lastT = currentT;
            currentT = nowT;
            if (gotData < 2)
                gotData++;
        }
        
        public float Interpolate() {
            interpolateT = Time.time;
            if (interpolateT == updateT || gotData < 2)
                return 0f;
            return Mathf.Min((interpolateT - updateT) / (float)(currentT - lastT), clamp);
        }
    }

    // This class is used to ensure max() based instead of add() based blendshape accumulation and to allow setting animator parameters from blendshape values
    private class OpenSeeBlendShapeProxy {
        private HashSet<string> browsBlendShapes = new HashSet<string>() { "BROWS UP", "BROWS DOWN", "BROWINNERUP", "BROWDOWNLEFT", "BROWDOWNRIGHT", "BROWOUTERUPLEFT", "BROWOUTERUPRIGHT", "NOSESNEERLEFT", "NOSESNEERRIGHT" };
        private HashSet<string> eyesBlendShapes = new HashSet<string>() { "BLINK", "BLINK_L", "BLINK_R", "EYELOOKUPLEFT", "EYELOOKUPRIGHT", "EYELOOKDOWNLEFT", "EYELOOKDOWNRIGHT", "EYELOOKINLEFT", "EYELOOKINRIGHT", "EYELOOKOUTLEFT", "EYELOOKOUTRIGHT", "EYEBLINKLEFT", "EYEBLINKRIGHT", "EYESQUINTRIGHT", "EYESQUINTLEFT", "EYEWIDELEFT", "EYEWIDERIGHT", "NOSESNEERLEFT", "NOSESNEERRIGHT", "CHEEKSQUINTLEFT", "CHEEKSQUINTRIGHT" };
        private HashSet<string> mouthBlendShapes = new HashSet<string>() { "A", "I", "U", "E", "O", "SIL", "CH", "DD", "FF", "KK", "NN", "PP", "RR", "SS", "TH", "JAWOPEN", "JAWFORWARD", "JAWLEFT", "JAWRIGHT", "MOUTHFUNNEL", "MOUTHPUCKER", "MOUTHLEFT", "MOUTHRIGHT", "MOUTHROLLUPPER", "MOUTHROLLLOWER", "MOUTHSHRUGUPPER", "MOUTHSHRUGLOWER", "MOUTHCLOSE", "MOUTHSMILELEFT", "MOUTHSMILERIGHT", "MOUTHFROWNLEFT", "MOUTHFROWNRIGHT", "MOUTHDIMPLELEFT", "MOUTHDIMPLERIGHT", "MOUTHUPPERUPLEFT", "MOUTHUPPERUPRIGHT", "MOUTHLOWERDOWNLEFT", "MOUTHLOWERDOWNRIGHT", "MOUTHPRESSLEFT", "MOUTHPRESSRIGHT", "MOUTHSTRETCHLEFT", "MOUTHSTRETCHRIGHT", "TONGUEOUT", "CHEEKSQUINTLEFT", "CHEEKSQUINTRIGHT" };
        private bool clearPSBrows = false;
        private bool clearPSEyes = false;
        private bool clearPSMouth = false;
        private Dictionary<BlendShapeKey, float> clearKeys = new Dictionary<BlendShapeKey, float>();
        
        private bool perfectSync = false;
        private HashSet<string> visemeBlendShapes = new HashSet<string>() { "A", "I", "U", "E", "O", "SIL", "CH", "DD", "FF", "KK", "NN", "PP", "RR", "SS", "TH" };
        private string[] perfectSyncNames = new string[] { "BROWINNERUP", "BROWDOWNLEFT", "BROWDOWNRIGHT", "BROWOUTERUPLEFT", "BROWOUTERUPRIGHT", "EYELOOKUPLEFT", "EYELOOKUPRIGHT", "EYELOOKDOWNLEFT", "EYELOOKDOWNRIGHT", "EYELOOKINLEFT", "EYELOOKINRIGHT", "EYELOOKOUTLEFT", "EYELOOKOUTRIGHT", "EYEBLINKLEFT", "EYEBLINKRIGHT", "EYESQUINTRIGHT", "EYESQUINTLEFT", "EYEWIDELEFT", "EYEWIDERIGHT", "CHEEKPUFF", "CHEEKSQUINTLEFT", "CHEEKSQUINTRIGHT", "NOSESNEERLEFT", "NOSESNEERRIGHT", "JAWOPEN", "JAWFORWARD", "JAWLEFT", "JAWRIGHT", "MOUTHFUNNEL", "MOUTHPUCKER", "MOUTHLEFT", "MOUTHRIGHT", "MOUTHROLLUPPER", "MOUTHROLLLOWER", "MOUTHSHRUGUPPER", "MOUTHSHRUGLOWER", "MOUTHCLOSE", "MOUTHSMILELEFT", "MOUTHSMILERIGHT", "MOUTHFROWNLEFT", "MOUTHFROWNRIGHT", "MOUTHDIMPLELEFT", "MOUTHDIMPLERIGHT", "MOUTHUPPERUPLEFT", "MOUTHUPPERUPRIGHT", "MOUTHLOWERDOWNLEFT", "MOUTHLOWERDOWNRIGHT", "MOUTHPRESSLEFT", "MOUTHPRESSRIGHT", "MOUTHSTRETCHLEFT", "MOUTHSTRETCHRIGHT", "TONGUEOUT" };
        private InterpolatedMap<string, InterpolatedFloat, float> perfectSyncMap = new InterpolatedMap<string, InterpolatedFloat, float>();
        private Dictionary<string, BlendShapeKey> clipMap = new Dictionary<string, BlendShapeKey>();

        private VRMBlendShapeProxy proxy = null;
        private Animator animator = null;
        private RuntimeAnimatorController animatorController = null;
        private Dictionary<BlendShapeKey, float> clearMap = new Dictionary<BlendShapeKey, float>();
        private Dictionary<BlendShapeKey, float> values = new Dictionary<BlendShapeKey, float>();
        private Dictionary<BlendShapeKey, string> clipNames = new Dictionary<BlendShapeKey, string>();
        private Dictionary<string, Tuple<string, AnimatorControllerParameterType>> parameters = new Dictionary<string, Tuple<string, AnimatorControllerParameterType>>();
        private bool interpolatedPerfectSync = false;
        private float defaultWeight = 1f;
        private OpenSeeVRMDriver vrmDriver = null;
        
        public OpenSeeBlendShapeProxy(OpenSeeVRMDriver vrmDriver) {
            this.vrmDriver = vrmDriver;
        }
        
        void InterpolatePerfectSync() {
            if (!HasPerfectSync() || interpolatedPerfectSync)
                return;
            foreach (var name in perfectSyncNames) {
                if (!perfectSyncMap.Check(name))
                    continue;
                float weight = perfectSyncMap.Get(name);
                AccumulateValue(name, weight, 1f);
            }
            interpolatedPerfectSync = true;
        }
        
        public void ClearPerfectSync() {
            perfectSyncMap.Clear();
        }
        
        public void SetPerfectSync(string name, float weight, double nowT, float factor) {
            perfectSyncMap.Store(name.ToUpper(), Mathf.Clamp(weight * factor, 0f, 1f), nowT);
        }

        public void SetPerfectSync(string name, float weight, double nowT) {
            SetPerfectSync(name, weight, nowT, defaultWeight);
        }
        
        public bool HasPerfectSync() {
            if (proxy == null || animator == null)
                return false;
            return perfectSync;
        }
        
        public void SetWeight(float v) {
            defaultWeight = v;
        }
        
        public void DisableFaceParts(bool brows, bool eyes, bool mouth) {
            clearPSBrows = brows;
            clearPSEyes = eyes;
            clearPSMouth = mouth;
        }
        
        private void SetFloat(BlendShapeKey key, float weight) {
            SetFloat(key, weight, defaultWeight);
        }
        
        private void SetFloat(BlendShapeKey key, float weight, float factor) {
            if (!clipNames.ContainsKey(key))
                return;
            string name = clipNames[key];
            weight *= factor;
            if (animator != null && parameters.ContainsKey(name)) {
                var parameter = parameters[name];
                if (parameter.Item2 == AnimatorControllerParameterType.Float)
                    animator.SetFloat(parameter.Item1, weight);
                if (parameter.Item2 == AnimatorControllerParameterType.Bool)
                    animator.SetBool(parameter.Item1, weight > 0.5f);
            }
        }

        private void CheckAnimatorController() {
            if (animator == null)
                return;
            if (animatorController != animator.runtimeAnimatorController) {
                animatorController = animator.runtimeAnimatorController;
                parameters.Clear();
                foreach (var parameter in animator.parameters) {
                    string key = parameter.name.ToUpper();
                    if (!parameters.ContainsKey(key) && (parameter.type == AnimatorControllerParameterType.Float || parameter.type == AnimatorControllerParameterType.Bool))
                        parameters.Add(key, new Tuple<string, AnimatorControllerParameterType>(parameter.name, parameter.type));
                    else
                        parameters[key] = new Tuple<string, AnimatorControllerParameterType>(parameter.name, parameter.type);
                }
            }
        }

        public void Clear() {
            CheckAnimatorController();
            interpolatedPerfectSync = false;
            if (proxy != null) {
                foreach (var pair in proxy.GetValues()) {
                    SetFloat(pair.Key, 0f);
                }
                foreach (var pair in values) {
                    SetFloat(pair.Key, 0f);
                }
                proxy.SetValues(clearMap);
            }
            values.Clear();
        }
        
        void NormalizeVisemes() {
            float total = 0f;
            foreach (var pair in values) {
                string name = pair.Key.Name.ToUpper();
                if (visemeBlendShapes.Contains(name))
                    total += pair.Value;
            }
            if (total > 1f) {
                var keys = new List<BlendShapeKey>(values.Keys);
                foreach (var key in keys) {
                    string name = key.Name.ToUpper();
                    if (visemeBlendShapes.Contains(name))
                        values[key] = values[key] / total;
                }
            }
        }
        
        public void Apply() {
            if (proxy == null)
                return;
            CheckAnimatorController();
            if (perfectSync)
                InterpolatePerfectSync();
            if (vrmDriver != null && vrmDriver.visemeNormalization)
                NormalizeVisemes();
            if (clearPSBrows || clearPSEyes || clearPSMouth) {
                clearKeys.Clear();
                foreach (var pair in proxy.GetValues()) {
                    string name = pair.Key.Name.ToUpper();
                    if (clearPSBrows && browsBlendShapes.Contains(name)) {
                        clearKeys[pair.Key] = 0f;
                        values[pair.Key] = 0f;
                    } else if (clearPSEyes && eyesBlendShapes.Contains(name)) {
                        clearKeys[pair.Key] = 0f;
                        values[pair.Key] = 0f;
                    } else if (clearPSMouth && mouthBlendShapes.Contains(name)) {
                        clearKeys[pair.Key] = 0f;
                        values[pair.Key] = 0f;
                    }
                }
                clearPSBrows = false;
                clearPSEyes = false;
                clearPSMouth = false;
            }
            proxy.SetValues(values);
            foreach (var pair in values) {
                SetFloat(pair.Key, pair.Value, 1f);
            }
            proxy.Apply();
        }

        public void AccumulateValue(BlendShapeKey key, float weight, float factor) {
            if (!values.ContainsKey(key))
                values.Add(key, weight * factor);
            else
                values[key] = Mathf.Max(values[key], weight * factor);
        }

        public void AccumulateValue(BlendShapeKey key, float weight) {
            AccumulateValue(key, weight, defaultWeight);
        }
        
        public void AccumulateValue(string key, float weight, float factor) {
            if (!clipMap.ContainsKey(key.ToUpper()))
                return;
            AccumulateValue(clipMap[key.ToUpper()], weight, factor);
        }
        
        public void AccumulateValue(string key, float weight) {
            AccumulateValue(key, weight, defaultWeight);
        }

        public void UpdateAvatar(VRMBlendShapeProxy vrmBlendShapeProxy, Animator animator) {
            proxy = vrmBlendShapeProxy;
            this.animator = animator;

            perfectSync = false;
            if (proxy == null)
                return;
            HashSet<string> perfectSyncSet = new HashSet<string>(perfectSyncNames);
            clipMap.Clear();
            clipNames.Clear();
            clearMap.Clear();
            perfectSyncMap.Clear();
            //perfectSyncMap.SetSmoothing(0.5f);
            foreach (BlendShapeClip clip in vrmBlendShapeProxy.BlendShapeAvatar.Clips) {
                if (clip.Preset == BlendShapePreset.Unknown && clip.BlendShapeName != null) {
                    string name = clip.BlendShapeName.ToUpper();
                    clipMap.Add(name, BlendShapeKey.CreateUnknown(clip.BlendShapeName));
                    clipNames.Add(BlendShapeKey.CreateUnknown(clip.BlendShapeName), name);
                    clearMap.Add(BlendShapeKey.CreateUnknown(clip.BlendShapeName), 0f);
                    if (perfectSyncSet.Contains(name)) {
                        perfectSyncSet.Remove(name);
                    }
                } else if (clip.Preset != BlendShapePreset.Unknown) {
                    string name = clip.BlendShapeName.ToUpper();
                    clipNames.Add(BlendShapeKey.CreateFromPreset(clip.Preset), name);
                    clearMap.Add(BlendShapeKey.CreateFromPreset(clip.Preset), 0f);
                }
            }
            if (perfectSyncSet.Count < 1)
                perfectSync = true;
        }
    }

    #if WINDOWS_BUILD
    private Dictionary<OVRLipSync.Viseme, float[]> catsData = null;
    #endif
    void InitCatsData() {
        #if WINDOWS_BUILD
        catsData = new Dictionary<OVRLipSync.Viseme, float[]>();
        // This is similar to what the Blender CATS plugin does, but with A, I, U, E, O, JawOpen
        catsData.Add(OVRLipSync.Viseme.sil, new float[]{0f, 0f, 0f, 0f, 0f, 0f});
        catsData.Add(OVRLipSync.Viseme.aa, new float[]{0.9998f, 0f, 0f, 0f, 0f, 1f});
        catsData.Add(OVRLipSync.Viseme.CH, new float[]{0f, 0.9996f, 0f, 0f, 0f, 1f});
        catsData.Add(OVRLipSync.Viseme.DD, new float[]{0.3f, 0.7f, 0f, 0f, 0f, 0.7f});
        catsData.Add(OVRLipSync.Viseme.E, new float[]{0f, 0f, 0f, 0.9997f, 0f, 0.6f});
        catsData.Add(OVRLipSync.Viseme.FF, new float[]{0.2f, 0.4f, 0f, 0f, 0f, 0.1f});
        catsData.Add(OVRLipSync.Viseme.ih, new float[]{0.5f, 0.2f, 0f, 0f, 0f, 0.5f});
        catsData.Add(OVRLipSync.Viseme.kk, new float[]{0.7f, 0.4f, 0f, 0f, 0f, 0.0f});
        catsData.Add(OVRLipSync.Viseme.nn, new float[]{0.2f, 0.7f, 0f, 0f, 0f, 0.1f});
        catsData.Add(OVRLipSync.Viseme.oh, new float[]{0f, 0f, 0f, 0f, 0.9999f, 1f});
        catsData.Add(OVRLipSync.Viseme.ou, new float[]{0f, 0f, 0.9995f, 0f, 0f, 1f});
        catsData.Add(OVRLipSync.Viseme.PP, new float[]{0.4f, 0f, 0f, 0f, 0.4f, 0f});
        catsData.Add(OVRLipSync.Viseme.RR, new float[]{0f, 0.5f, 0f, 0f, 0.3f, 0.4f});
        catsData.Add(OVRLipSync.Viseme.SS, new float[]{0f, 0.8f, 0f, 0f, 0f, 0.3f});
        catsData.Add(OVRLipSync.Viseme.TH, new float[]{0.4f, 0f, 0f, 0f, 0.15f, 0.5f});
        #endif
        visemePresetMap = new BlendShapeKey[6] {
            BlendShapeKey.CreateFromPreset(BlendShapePreset.A),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.I),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.U),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.E),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.O),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.Unknown)
        };
        lastVisemeValues = new float[] {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f};
        #if WINDOWS_BUILD
        ovrMap = new Dictionary<OVRLipSync.Viseme, int>() {
            [OVRLipSync.Viseme.sil] = 0,
            [OVRLipSync.Viseme.aa] = 1,
            [OVRLipSync.Viseme.CH] = 2,
            [OVRLipSync.Viseme.DD] = 3,
            [OVRLipSync.Viseme.E] = 4,
            [OVRLipSync.Viseme.FF] = 5,
            [OVRLipSync.Viseme.ih] = 6,
            [OVRLipSync.Viseme.kk] = 7,
            [OVRLipSync.Viseme.nn] = 8,
            [OVRLipSync.Viseme.oh] = 9,
            [OVRLipSync.Viseme.ou] = 10,
            [OVRLipSync.Viseme.PP] = 11,
            [OVRLipSync.Viseme.RR] = 12,
            [OVRLipSync.Viseme.SS] = 13,
            [OVRLipSync.Viseme.TH] = 14
        };
        #endif
        fullVisemePresets = new BlendShapeKey[16] {
            BlendShapeKey.CreateUnknown("SIL"),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.A),
            BlendShapeKey.CreateUnknown("CH"),
            BlendShapeKey.CreateUnknown("DD"),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.E),
            BlendShapeKey.CreateUnknown("FF"),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.I),
            BlendShapeKey.CreateUnknown("KK"),
            BlendShapeKey.CreateUnknown("NN"),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.O),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.U),
            BlendShapeKey.CreateUnknown("PP"),
            BlendShapeKey.CreateUnknown("RR"),
            BlendShapeKey.CreateUnknown("SS"),
            BlendShapeKey.CreateUnknown("TH"),
            BlendShapeKey.CreateFromPreset(BlendShapePreset.Unknown)
        };
    }

    #if WINDOWS_BUILD
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
    #endif
    
    bool ApplyVisemes() {
        #if WINDOWS_BUILD
        trackMouth = false;
        if (vrmBlendShapeProxy == null || catsData == null)
            return true;
        float expressionFactor = 1f;
        if (currentExpression != null)
            expressionFactor = currentExpressionVisemeFactor;
        if (currentExpression != null && !currentExpressionEnableVisemes) {
            return true;
        }
        float weight;
        OVRLipSync.Viseme current = GetActiveViseme(out weight);
        if (current == OVRLipSync.Viseme.sil) {
            if (wasSilViseme) {
                if (hybridLipSync && Time.time - startedSilVisemes > silVisemeHybridThreshold) {
                    ApplyMouthShape();
                    return false;
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
        if (!maximizeLipSync)
            weight = Mathf.Clamp(weight * 1.5f, 0f, 1f);
        else
            weight = 1f;

        float[] values = new float[16];
        int lastIdx = 5;
        if (haveFullVisemeSet) {
            lastIdx = 15;
            values[ovrMap[current]] = weight;
            values[lastIdx] = catsData[current][5];
        } else {
            Array.Copy(catsData[current], values, 6);
        }

        for (int i = 0; i <= lastIdx; i++) {
            lastVisemeValues[i] = values[i] * weight * (1f - visemeSmoothing) + lastVisemeValues[i] * visemeSmoothing;
            if (lastVisemeValues[i] < 0f) {
                lastVisemeValues[i] = 0f;
            }
            if (lastVisemeValues[i] > 1f) {
                lastVisemeValues[i] = 1f;
            }
            float result = lastVisemeValues[i] * visemeFactor * expressionFactor;
            if (result > 0f && i < lastIdx) {
                if (haveFullVisemeSet)
                    proxy.AccumulateValue(fullVisemePresets[i], result, 1f);
                else
                    proxy.AccumulateValue(visemePresetMap[i], result, 1f);
            }
            if (i == lastIdx) {
                if (animateJawBone)
                    SetJaw(result * 0.99999f);
                else
                    SetJaw(0f);
            }
        }
        #endif
        return false;
    }
    
    void SetJaw(float v) {
        if (haveJawParameter) {
            animator.SetFloat("JawMovement", v);
            return;
        }
        if (jawBoneAnimation != null && jawBone != null) {
            for (int i = 0; i < humanBodyBones.Length; i++) {
                if (humanBodyBoneTransforms[i] != null)
                    humanBodyBoneRotations[i] = humanBodyBoneTransforms[i].localRotation;
            }
            jawBoneAnimation.SampleAnimation(lastAvatar.gameObject, v / jawBoneAnimation.frameRate);
            jawRotation = jawBone.localRotation;
            for (int i = 0; i < humanBodyBones.Length; i++) {
                if (humanBodyBoneTransforms[i] != null)
                    humanBodyBoneTransforms[i].localRotation = humanBodyBoneRotations[i];
            }
            return;
        }
    }
    
    void ApplyMouthShape(bool fadeOnly) {
        if (vrmBlendShapeProxy == null)
            return;
        if (visemePresetMap == null)
            InitCatsData();
        if (openSeeData == null || openSeeData.features == null)
            return;
        
        if (lastMouthStates == null)
            lastMouthStates = new float[]{0f, 0f, 0f, 0f, 0f, 0f};
        if (currentMouthStates == null)
            currentMouthStates = new float[]{0f, 0f, 0f, 0f, 0f, 0f};
        
        if (mouthInterpolate == null)
            mouthInterpolate = new TimeInterpolate();
        float t = mouthInterpolate.Interpolate();
        
        if (currentExpression != null && !currentExpressionEnableVisemes)
            return;
        
        trackMouth = !fadeOnly;
        
        float expressionFactor = 1f;
        if (currentExpression != null)
            expressionFactor = currentExpressionVisemeFactor;

        for (int i = 0; i < 6; i++) {
            float interpolated = Mathf.Lerp(lastMouthStates[i], currentMouthStates[i], t);
            float result = interpolated * expressionFactor * visemeFactor;
            if (i < 5 && result > 0f)
                if (!useCameraPerfectSync || !proxy.HasPerfectSync())
                    proxy.AccumulateValue(visemePresetMap[i], result);
            if (i == 5) {
                if (animateJawBone)
                    SetJaw(result * 0.99999f * blendShapeWeight);
                else
                    SetJaw(0f);
            }
        }

        if (lastMouth < openSeeData.time) {
            lastMouth = openSeeData.time;
            mouthInterpolate.UpdateTime(lastMouth);
            float open = openSeeData.features.MouthOpen;
            float wide = openSeeData.features.MouthWide;
            float[] mouthStates = new float[]{0f, 0f, 0f, 0f, 0f, 0f};
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
                mouthStates[5] = Mathf.Clamp(open / 0.55f - 0.1f, 0f, 1f);
                if (open < stabilizer && Mathf.Abs(wide) < stabilizer)
                    break;
                if (wide > stabilizer && open < stabilizerWide)
                    break;
                if (open > 0.5f) {
                    // O
                    mouthStates[4] = open;
                } else if (open >= 0f) {
                    // A
                    mouthStates[0] = open;
                }
                if (wide >= 0f && open > stabilizer * 0.5f) {
                    if (wide > 0.5f) {
                        // I
                        mouthStates[1] = wide;
                    } else {
                        // E
                        mouthStates[3] = wide;
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
            
            for (int i = 0; i < 6; i++) {
                lastMouthStates[i] = Mathf.Lerp(lastMouthStates[i], currentMouthStates[i], t);
                mouthStates[i] = Mathf.Lerp(lastMouthStates[i], mouthStates[i], 1f - mouthSmoothing);
            }
            currentMouthStates = mouthStates;
        }
    }
    
    void ApplyMouthShape() {
        ApplyMouthShape(false);
    }
    
    void UpdatePerfectSync() {
        if (openSeeData == null || lastPerfectSync >= openSeeData.time)
            return;
        lastPerfectSync = openSeeData.time;
        if (useCameraPerfectSync && proxy.HasPerfectSync()) {
            float upDownStrength = (openSeeData.features.EyebrowUpDownLeft + openSeeData.features.EyebrowUpDownRight) / 2f;
            if (upDownStrength > 0f)
                proxy.SetPerfectSync("BROWINNERUP", upDownStrength * 0.8f, lastPerfectSync);
            else
                proxy.SetPerfectSync("BROWINNERUP", 0, lastPerfectSync);

            if (openSeeData.features.EyebrowUpDownLeft < 0.2f) {
                proxy.SetPerfectSync("BROWDOWNLEFT", -openSeeData.features.EyebrowUpDownLeft * 0.5f, lastPerfectSync);
                if (openSeeData.features.EyeLeft < 0.1f && openSeeData.features.EyeLeft > -0.6f) {
                    proxy.SetPerfectSync("EYESQUINTLEFT", -openSeeData.features.EyeLeft, lastPerfectSync);
                    proxy.SetPerfectSync("EYEBLINKLEFT", 0, lastPerfectSync);
                } else if (openSeeData.features.EyeLeft <= -0.6f) {
                    proxy.SetPerfectSync("EYEBLINKLEFT", -openSeeData.features.EyeLeft * 1.5f, lastPerfectSync);
                    proxy.SetPerfectSync("EYESQUINTLEFT", 0, lastPerfectSync);
                } else {
                    proxy.SetPerfectSync("EYESQUINTLEFT", 0, lastPerfectSync);
                    proxy.SetPerfectSync("EYEBLINKLEFT", 0, lastPerfectSync);
                }
            } else {
                proxy.SetPerfectSync("BROWDOWNLEFT", 0, lastPerfectSync);
                proxy.SetPerfectSync("EYESQUINTLEFT", 0, lastPerfectSync);
                if (openSeeData.features.EyeLeft <= -0.3f)
                    proxy.SetPerfectSync("EYEBLINKLEFT", -openSeeData.features.EyeLeft * 1.5f, lastPerfectSync);
                else
                    proxy.SetPerfectSync("EYEBLINKLEFT", 0, lastPerfectSync);
            }
            
            if (openSeeData.features.EyebrowUpDownRight < 0.2f) {
                proxy.SetPerfectSync("BROWDOWNRIGHT", -openSeeData.features.EyebrowUpDownRight * 0.5f, lastPerfectSync);
                if (openSeeData.features.EyeRight < 0.1f && openSeeData.features.EyeRight > -0.6f) {
                    proxy.SetPerfectSync("EYESQUINTRIGHT", -openSeeData.features.EyeRight, lastPerfectSync);
                    proxy.SetPerfectSync("EYEBLINKRIGHT", 0, lastPerfectSync);
                } else if (openSeeData.features.EyeRight <= -0.6f) {
                    proxy.SetPerfectSync("EYEBLINKRIGHT", -openSeeData.features.EyeRight * 1.5f, lastPerfectSync);
                    proxy.SetPerfectSync("EYESQUINTRIGHT", 0, lastPerfectSync);
                } else {
                    proxy.SetPerfectSync("EYESQUINTRIGHT", 0, lastPerfectSync);
                    proxy.SetPerfectSync("EYEBLINKRIGHT", 0, lastPerfectSync);
                }
            } else {
                proxy.SetPerfectSync("BROWDOWNRIGHT", 0, lastPerfectSync);
                proxy.SetPerfectSync("EYESQUINTRIGHT", 0, lastPerfectSync);
                if (openSeeData.features.EyeRight <= -0.3f)
                    proxy.SetPerfectSync("EYEBLINKRIGHT", -openSeeData.features.EyeRight * 1.5f, lastPerfectSync);
                else
                    proxy.SetPerfectSync("EYEBLINKRIGHT", 0, lastPerfectSync);
            }

            if (openSeeData.features.EyebrowSteepnessLeft < 0.2f)
                proxy.SetPerfectSync("BROWOUTERUPLEFT", -openSeeData.features.EyebrowSteepnessLeft, lastPerfectSync);
            else
                proxy.SetPerfectSync("BROWOUTERUPLEFT", 0, lastPerfectSync);
            
            if (openSeeData.features.EyebrowSteepnessRight < 0.2f)
                proxy.SetPerfectSync("BROWOUTERUPRIGHT", -openSeeData.features.EyebrowSteepnessRight, lastPerfectSync);
            else
                proxy.SetPerfectSync("BROWOUTERUPRIGHT", 0, lastPerfectSync);

            if (openSeeData.features.EyeLeft > 0.5f)
                proxy.SetPerfectSync("EYEWIDELEFT", openSeeData.features.EyeLeft * 0.7f, lastPerfectSync);
            else
                proxy.SetPerfectSync("EYEWIDELEFT", 0, lastPerfectSync);
            
            if (openSeeData.features.EyeRight > 0.5f)
                proxy.SetPerfectSync("EYEWIDERIGHT", openSeeData.features.EyeRight * 0.7f, lastPerfectSync);
            else
                proxy.SetPerfectSync("EYEWIDERIGHT", 0, lastPerfectSync);
            
            if (trackMouth) {
                if (openSeeData.features.MouthWide < 0.5f)
                    proxy.SetPerfectSync("MOUTHPUCKER", -openSeeData.features.MouthWide * 0.3f, lastPerfectSync);
                else
                    proxy.SetPerfectSync("MOUTHPUCKER", 0, lastPerfectSync);
                
                if (openSeeData.features.MouthOpen > 0.0f) {
                    proxy.SetPerfectSync("JAWOPEN", openSeeData.features.MouthOpen, lastPerfectSync);
                    proxy.SetPerfectSync("MOUTHCLOSE", 0, lastPerfectSync);
                } else if (openSeeData.features.MouthOpen < -0.0f) {
                    proxy.SetPerfectSync("MOUTHCLOSE", openSeeData.features.MouthOpen, lastPerfectSync);
                    proxy.SetPerfectSync("JAWOPEN", 0, lastPerfectSync);
                } else {
                    proxy.SetPerfectSync("JAWOPEN", 0, lastPerfectSync);
                    proxy.SetPerfectSync("MOUTHCLOSE", 0, lastPerfectSync);
                }
                
                if (openSeeData.features.MouthCornerInOutLeft > 0.3f)
                    proxy.SetPerfectSync("MOUTHLEFT", openSeeData.features.MouthCornerInOutLeft * 0.5f, lastPerfectSync);
                else
                    proxy.SetPerfectSync("MOUTHLEFT", 0, lastPerfectSync);

                if (openSeeData.features.MouthCornerInOutRight > 0.3f)
                    proxy.SetPerfectSync("MOUTHRIGHT", openSeeData.features.MouthCornerInOutRight * 0.5f, lastPerfectSync);
                else
                    proxy.SetPerfectSync("MOUTHRIGHT", 0, lastPerfectSync);

                if (openSeeData.features.MouthCornerUpDownLeft > 0.3f)
                    proxy.SetPerfectSync("MOUTHSMILELEFT", openSeeData.features.MouthCornerUpDownLeft * 0.5f, lastPerfectSync);
                else if (openSeeData.features.MouthCornerUpDownLeft < -0.3f)
                    proxy.SetPerfectSync("MOUTHFROWNLEFT", -openSeeData.features.MouthCornerUpDownLeft, lastPerfectSync);
                else {
                    proxy.SetPerfectSync("MOUTHSMILELEFT", 0, lastPerfectSync);
                    proxy.SetPerfectSync("MOUTHFROWNLEFT", 0, lastPerfectSync);
                }

                if (openSeeData.features.MouthCornerUpDownRight > 0.3f)
                    proxy.SetPerfectSync("MOUTHSMILERIGHT", openSeeData.features.MouthCornerUpDownRight * 0.5f, lastPerfectSync);
                else if (openSeeData.features.MouthCornerUpDownLeft < -0.3f)
                    proxy.SetPerfectSync("MOUTHFROWNRIGHT", -openSeeData.features.MouthCornerUpDownRight, lastPerfectSync);
                else {
                    proxy.SetPerfectSync("MOUTHSMILERIGHT", 0, lastPerfectSync);
                    proxy.SetPerfectSync("MOUTHFROWNRIGHT", 0, lastPerfectSync);
                }
            }
        }
    }
    
    void UpdateBrows() {
        if ((browClips == null && (faceMesh == null || faceType < 0)) || openSeeData == null) {
            return;
        }
        if (browInterpolate == null)
            browInterpolate = new TimeInterpolate();

        float t = browInterpolate.Interpolate();
        float strength = eyebrowStrength;
        if (currentExpression != null)
            strength *= currentExpressionEyebrowWeight;
        
        if (!useCameraPerfectSync || !proxy.HasPerfectSync()) {
            if (!skipApply && browClips == null) {
                if (browUpIndex > -1 && strength > 0f)
                    faceMesh.SetBlendShapeWeight(browUpIndex, strength * Mathf.Lerp(lastBrowStates[0], currentBrowStates[0], t));
                if (browDownIndex > -1 && strength > 0f)
                    faceMesh.SetBlendShapeWeight(browDownIndex, strength * Mathf.Lerp(lastBrowStates[1], currentBrowStates[1], t));
                if (browAngryIndex > -1 && strength > 0f)
                    faceMesh.SetBlendShapeWeight(browAngryIndex, strength * Mathf.Lerp(lastBrowStates[2], currentBrowStates[2], t));
                if (browSorrowIndex > -1 && strength > 0f)
                    faceMesh.SetBlendShapeWeight(browSorrowIndex, strength * Mathf.Lerp(lastBrowStates[3], currentBrowStates[3], t));
            }
            if (browClips != null && strength > 0f) {
                if (lastBrowUpDown > 0f)
                    proxy.AccumulateValue(browClips[0], strength * Mathf.Lerp(lastBrowStates[0], currentBrowStates[0], t));
                else if (lastBrowUpDown < 0f)
                    proxy.AccumulateValue(browClips[1], strength * Mathf.Lerp(lastBrowStates[1], currentBrowStates[1], t));
            }
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
            upDownStrength = Mathf.LerpUnclamped(lastBrowUpDown, upDownStrength, 1f - eyebrowSmoothing);
            lastBrowUpDown = upDownStrength;
            
            for (int i = 0; i < 4; i++)
                lastBrowStates[i] = Mathf.Lerp(lastBrowStates[i], currentBrowStates[i], t);
            browInterpolate.UpdateTime(openSeeData.time);
            
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
    }
    
    void FindFaceMesh() {
        if (lastAvatar == vrmBlendShapeProxy)
            return;
        if (visemePresetMap == null)
            InitCatsData();
        lastAvatar = vrmBlendShapeProxy;
        animator = lastAvatar.gameObject.GetComponent<Animator>();

        if (animator != null) {
            haveJawParameter = false;
            for (int i = 0; i < animator.parameterCount; i++) {
                if (animator.parameters[i].name == "JawMovement") {
                    haveJawParameter = true;
                    break;
                }
            }
            jawBone = animator.GetBoneTransform(HumanBodyBones.Jaw);
            for (int i = 0; i < humanBodyBones.Length; i++) {
                if (humanBodyBones[i] >= 0 && humanBodyBones[i] < HumanBodyBones.LastBone)
                    humanBodyBoneTransforms[i] = animator.GetBoneTransform(humanBodyBones[i]);
                else
                    humanBodyBoneTransforms[i] = null;
            }
        }
        
        proxy.UpdateAvatar(lastAvatar, animator);

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
        
        HashSet<string> visemeClipNames = new HashSet<string>();
        foreach (var key in fullVisemePresets)
            if (key.Preset == BlendShapePreset.Unknown && key.Name != "")
                visemeClipNames.Add(key.Name.ToUpper());

        // Check if the model has set up blendshape clips. Thanks do Deat for this idea!
        string foundClipUp = null;
        string foundClipDown = null;
        HashSet<string> foundVisemeClips = new HashSet<string>();
        foreach (BlendShapeClip clip in vrmBlendShapeProxy.BlendShapeAvatar.Clips) {
            if (clip.Preset == BlendShapePreset.Unknown && clip.BlendShapeName != null) {
                string name = clip.BlendShapeName.ToUpper();
                if (name == "BROWS UP")
                    foundClipUp = clip.BlendShapeName;
                if (name == "BROWS DOWN")
                    foundClipDown = clip.BlendShapeName;
                if (visemeClipNames.Contains(name))
                    foundVisemeClips.Add(name);
            }
        }
        haveFullVisemeSet = false;
        if (visemeClipNames.SetEquals(foundVisemeClips)) {
            Debug.Log("Detected full viseme set on avatar.");
            haveFullVisemeSet = true;
        }
        if (foundClipUp != null && foundClipDown != null) {
            browClips =  new BlendShapeKey[2] {
                BlendShapeKey.CreateUnknown(foundClipUp),
                BlendShapeKey.CreateUnknown(foundClipDown)
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
            if (triggered != lastActive)
                reached = false;
            if (reached) {
                if (triggered) {
                    lastWeight = 1f;
                    return 1f;
                } else {
                    lastWeight = 0f;
                    return 0f;
                }
            }
            if (triggered != lastActive) {
                stateChangedTime = Time.time;
                stateChangedWeight = lastWeight;
                currentTransitionTime = transitionTime;
                lastActive = triggered;
                if (triggered) {
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
            if (lastWeight >= 0.9999f && triggered) {
                lastWeight = 1f;
                reached = true;
            }
            if (lastWeight <= 0.0001f && !triggered) {
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
        lastExpression = null;
        currentExpression = null;
        overridden = false;
        continuedPress = new HashSet<OpenSeeVRMExpression>();
        expressionMap = new Dictionary<string, OpenSeeVRMExpression>();
        openSeeExpression.weightMap = new Dictionary<string, float>();
        foreach (var expression in expressions) {
            expression.toggled = false;
            expression.triggered = false;
            expression.lastActive = false;
            expression.reached = true;
            openSeeExpression.weightMap.Add(expression.trigger, expression.errorWeight);
            if (expression.customBlendShapeName != "")
                expression.blendShapeKey = BlendShapeKey.CreateUnknown(expression.customBlendShapeName);
            else
                expression.blendShapeKey = BlendShapeKey.CreateFromPreset(expression.blendShapePreset);
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
                // Clean up inconsistent states
                if (!expression.toggle && expression.toggled) {
                    expression.toggled = false;
                    if (toggledExpression == expression) {
                        toggledExpression = null;
                        overridden = false;
                    }
                    if (continuedPress.Contains(expression))
                        continuedPress.Remove(expression);
                }
                if (expression.toggled && expression.additive && toggledExpression == expression) {
                    toggledExpression = null;
                    overridden = false;
                }
                if (expression.toggled && !expression.additive && toggledExpression != expression) {
                    if (toggledExpression == null && !overridden) {
                        toggledExpression = expression;
                        overridden = true;
                    } else
                        expression.toggled = false;
                }
                
                // Automatically trigger toggled expressions
                if ((expression.additive || expression == toggledExpression) && expression.toggled)
                    expression.triggered = true;
                else
                    expression.triggered = false;
                
                if (expression.hotkey >= 0 && expression.hotkey < 256 && (GetAsyncKeyState(expression.hotkey) & 0x8000) != 0 && shiftKey == expression.shiftKey && ctrlKey == expression.ctrlKey && altKey == expression.altKey) {
                    if (!expression.toggle && continuedPress.Contains(expression))
                        continuedPress.Remove(expression);
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
                        } else {
                            if (!expression.additive) {
                                // If it's a base expression being triggered on, override expression detection
                                trigger = true;
                                currentExpression = expression;
                            }
                        }
                    }
                } else {
                    continuedPress.Remove(expression);
                }
            }
        }

        if (overridden && !trigger)
            currentExpression = toggledExpression;
        if (trigger && overridden && currentExpression != toggledExpression)
            toggledExpression.triggered = false;
        
        if (!overridden && !trigger) {
            if (only30Points || !openSeeExpression.enabled || openSeeExpression.expressionTime + 15f < Time.time || openSeeExpression.expression == null || openSeeExpression.expression == "" || !expressionMap.ContainsKey(openSeeExpression.expression)) {
                currentExpression = null;
            } else {
                currentExpression = expressionMap[openSeeExpression.expression];
                currentExpression.triggered = true;
            }
        } else if (!only30Points && openSeeExpression.enabled && openSeeExpression.expressionTime + 15f > Time.time && openSeeExpression.expression != null && openSeeExpression.expression != "" && expressionMap.ContainsKey(openSeeExpression.expression)) {
            if (expressionMap[openSeeExpression.expression].additive)
                expressionMap[openSeeExpression.expression].triggered = true;
        }
        
        if (currentExpression == null && expressionMap.ContainsKey("neutral")) {
            currentExpression = expressionMap["neutral"];
            currentExpression.triggered = true;
        }

        // Reset limits on expressions
        currentExpressionEnableBlinking = true;
        currentExpressionEnableVisemes = true;
        currentExpressionEyebrowWeight = 1f;
        currentExpressionVisemeFactor = 1f;

        float baseTransitionTime = 1000f;
        if (currentExpression != null && (!currentExpression.reached || currentExpression.lastActive != currentExpression.triggered) && currentExpression.transitionTime < baseTransitionTime)
            baseTransitionTime = currentExpression.transitionTime;
        if (lastExpression != currentExpression && lastExpression != null && (!lastExpression.reached || lastExpression.lastActive != lastExpression.triggered) && lastExpression.transitionTime < baseTransitionTime)
            baseTransitionTime = lastExpression.transitionTime;
        foreach (var expression in expressionMap.Values) {
            float weight = expression.GetWeight(baseTransitionTime);
            if (weight > 0f) {
                proxy.AccumulateValue(expression.blendShapeKey, weight, 1f);
                
                // Accumulate limits on expressions
                currentExpressionEnableBlinking = currentExpressionEnableBlinking && expression.enableBlinking;
                currentExpressionEnableVisemes = currentExpressionEnableVisemes && expression.enableVisemes;
                currentExpressionEyebrowWeight = Mathf.Min(currentExpressionEyebrowWeight, expression.eyebrowWeight);
                currentExpressionVisemeFactor = Mathf.Min(currentExpressionVisemeFactor, expression.visemeFactor);
            }
        }
        proxy.DisableFaceParts(currentExpressionEyebrowWeight < 0.0005f, !currentExpressionEnableBlinking, !currentExpressionEnableVisemes || (currentExpressionVisemeFactor < 0.0005f));
        lastExpression = currentExpression;
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
        if (openSeeData == null || vrmBlendShapeProxy == null || openSeeIKTarget == null || skipApplyEyes)
            return;
        
        float lookUpDown = currentLookUpDown;
        float lookLeftRight = currentLookLeftRight;
        
        if (gazeInterpolate == null)
            gazeInterpolate = new TimeInterpolate();
        float t = gazeInterpolate.Interpolate();

        lookLeftRight = Mathf.Lerp(lastLookLeftRight, currentLookLeftRight, t);
        lookUpDown = Mathf.Lerp(lastLookUpDown, currentLookUpDown, t);
        
        if (!openSeeIKTarget.mirrorMotion)
            lookLeftRight = -lookLeftRight;
        
        if (leftEye == null && rightEye == null) {
            if (gazeTracking) {
                if (lookUpDown > 0f) {
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.LookUp), gazeFactor.x * lookUpDown);
                } else {
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.LookDown), -gazeFactor.x * lookUpDown);
                }
                
                if (lookLeftRight > 0f) {
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.LookLeft), gazeFactor.y * lookLeftRight);
                } else {
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.LookRight), -gazeFactor.y * lookLeftRight);
                }
            }
        } else {
            if (rightEye != null)
                rightEye.localRotation = Quaternion.identity;
            if (leftEye != null)
                leftEye.localRotation = Quaternion.identity;
            if (gazeTracking) {
                Quaternion rotation = Quaternion.Slerp(Quaternion.identity, Quaternion.AngleAxis(-gazeFactor.x * gazeStrength * lookUpDown, Vector3.right) * Quaternion.AngleAxis(-gazeFactor.y * gazeStrength * lookLeftRight, Vector3.up), blendShapeWeight);
                if (rightEye != null)
                    rightEye.localRotation = rotation;
                if (leftEye != null)
                    leftEye.localRotation = rotation;
            }
        }

        if (lastGaze < openSeeData.time) {
            GetLookParameters(ref lookLeftRight, ref lookUpDown, false, 66, 36, 39, 37, 38, 41, 40);
            GetLookParameters(ref lookLeftRight, ref lookUpDown, true,  67, 42, 45, 43, 44, 47, 46);
            
            lookLeftRight = Mathf.Lerp(currentLookLeftRight, lookLeftRight, 1f - gazeSmoothing);
            lookUpDown = Mathf.Lerp(currentLookUpDown, lookUpDown, 1f - gazeSmoothing);

            if (Mathf.Abs(lookLeftRight - currentLookLeftRight) + Mathf.Abs(lookUpDown - currentLookUpDown) > gazeStabilizer) {
                lastLookLeftRight = Mathf.Lerp(lastLookLeftRight, currentLookLeftRight, t);
                lastLookUpDown = Mathf.Lerp(lastLookUpDown, currentLookUpDown, t);
                currentLookLeftRight = lookLeftRight;
                currentLookUpDown = lookUpDown;
                lastGaze = openSeeData.time;
                gazeInterpolate.UpdateTime(lastGaze);
            }
        }
    }

    void BlinkEyes() {
        if (vrmBlendShapeProxy == null || eyeBlinker == null)
            return;
        if (autoBlink || only30Points) {
            if (currentExpressionEnableBlinking) {
                float blink = eyeBlinker.Blink();
                if (!useCameraPerfectSync || ! proxy.HasPerfectSync())
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink), blink, 1);
            }
        } else if (openSeeExpression != null && openSeeExpression.openSee != null && openSeeData != null) {
            if (!currentExpressionEnableBlinking)
                return;

            float right = 1f;
            float left = 1f;
            
            float upDownAngle = openSeeData.rawEuler.x;
            while (upDownAngle < 0f)
                upDownAngle += 360f;
            while (upDownAngle >= 360f)
                upDownAngle -= 360f;
            if (lookDownCompensation && upDownAngle > turnDownBoundaryAngle - lookDownAdjustment - wasLookingDown && autoBlinkLookingDown) {
                if (!(wasLookingDown > 0f))
                    eyeBlinker.Blink();
                float blink = eyeBlinker.Blink();
                if (!useCameraPerfectSync || ! proxy.HasPerfectSync())
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink), blink);
                return;
            }

            if (blinkInterpolate == null)
                blinkInterpolate = new TimeInterpolate();
            float t = blinkInterpolate.Interpolate();
            
            left = Mathf.Lerp(lastBlinkLeft, currentBlinkLeft, t);
            right = Mathf.Lerp(lastBlinkRight, currentBlinkRight, t);
            
            if (openSeeIKTarget.mirrorMotion) {
                float tmp = left;
                left = right;
                right = tmp;
            }
            
            if (!useCameraPerfectSync || !proxy.HasPerfectSync()) {
                if (linkBlinks && !allowWinking) {
                    float v = Mathf.Max(left, right);
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink), v);
                } else {
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink_R), right);
                    proxy.AccumulateValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink_L), left);
                }
            }

            if (openSeeData != null && lastBlink < openSeeData.time) {
                lastBlink = openSeeData.time;
                blinkInterpolate.UpdateTime(lastBlink);
                float openThreshold = eyeOpenedThreshold;
                float closedThreshold = eyeClosedThreshold;
                
                if (openSeeData.rawEuler.y < turnLeftBoundaryAngle || openSeeData.rawEuler.y > turnRightBoundaryAngle)
                    openThreshold = Mathf.Lerp(eyeOpenedThreshold, eyeClosedThreshold, 0.4f);

                if (lookDownCompensation && upDownAngle > turnDownBoundaryAngle - lookDownAdjustment - wasLookingDown) {
                    openThreshold = Mathf.Lerp(eyeOpenedThreshold, eyeClosedThreshold, 0.85f);
                    closedThreshold = Mathf.Lerp(eyeClosedThreshold, 0f, 0.6f);
                    if (upDownAngle > turnDownBoundaryAngle - lookDownAdjustment + 10f) {
                        openThreshold = 0.1f;
                        closedThreshold = 0f;
                    }
                    wasLookingDown = 1.5f;
                } else {
                    wasLookingDown = 0f;
                }
                
                if (openSeeData.rightEyeOpen > openThreshold)
                    right = 0f;
                else if (openSeeData.rightEyeOpen < closedThreshold)
                    right = 1f;
                else
                    right = 1f - (openSeeData.rightEyeOpen - closedThreshold) / (openThreshold - closedThreshold);

                if (openSeeData.leftEyeOpen > openThreshold)
                    left = 0f;
                else if (openSeeData.leftEyeOpen < closedThreshold)
                    left = 1f;
                else
                    left = 1f - (openSeeData.leftEyeOpen - closedThreshold) / (openThreshold - closedThreshold);
                
                if (openSeeData.rawEuler.y < turnLeftBoundaryAngle)
                    left = right;
                if (openSeeData.rawEuler.y > turnRightBoundaryAngle)
                    right = left;
                
                if (linkBlinks) {
                    if (Mathf.Abs(right - left) < smartWinkThreshold || !allowWinking) {
                        float v = Mathf.Max(left, right);
                        if (v < 0.3f)
                            v = Mathf.Min(left, right);
                        left = v;
                        right = v;
                    }
                }

                lastBlinkLeft = Mathf.Lerp(lastBlinkLeft, currentBlinkLeft, t);
                lastBlinkRight = Mathf.Lerp(lastBlinkRight, currentBlinkRight, t);
                currentBlinkLeft = Mathf.Lerp(currentBlinkLeft, left, 1f - blinkSmoothing);
                currentBlinkRight = Mathf.Lerp(currentBlinkRight, right, 1f - blinkSmoothing);
                
                if (left < 0.00001f)
                    currentBlinkLeft = 0f;
                if (right < 0.00001f)
                    currentBlinkRight = 0f;
                if (left > 0.99999f)
                    currentBlinkLeft = 1f;
                if (right > 0.99999f)
                    currentBlinkRight = 1f;
            }
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
        #if WINDOWS_BUILD
        active = false;
        lastAudioTime = -1f;
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
        try { Microphone.GetDeviceCaps(lastMic, out minFreq, out maxFreq); } catch (Exception e) { Debug.LogError("Failed to get device capabilities: " + e); }
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

        try { clip = Microphone.Start(lastMic, true, 1, freq); } catch (Exception e) { Debug.LogError("Failed to start microphone: " + e); lipSync = false;}
        channels = clip.channels;
        partialAudio = new float[1024 * channels];
        lastPos = 0;
        active = true;
        #endif
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
        #if WINDOWS_BUILD
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
        
        if (audioVolume > 0f)
            lastAudioTime = fakeTimeTime;

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
        #endif
    }

    void OnAudioFilterRead(float[] data, int channels) {
        #if WINDOWS_BUILD
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
        #endif
    }

    void ReadAudio() {
        #if WINDOWS_BUILD
        if (lastAudioTime >= 0f && Time.time > lastAudioTime + 3f && lipSync) {
            Debug.Log("Audio device might have disappeared, reinitializing lip sync.");
            InitializeLipSync();
        }
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
        #endif
    }

    void CreateContext() {
        #if WINDOWS_BUILD
        lock (this) {
            if (context == 0) {
                if (OVRLipSync.CreateContext(ref context, provider) != OVRLipSync.Result.Success) {
                    Debug.LogError("OVRLipSyncContextBase.Start ERROR: Could not create Phoneme context.");
                    return;
                }
            }
        }
        #endif
    }

    void Start() {
        InitExpressionMap();
        proxy = new OpenSeeBlendShapeProxy(this);
        humanBodyBones = (HumanBodyBones[])Enum.GetValues(typeof(HumanBodyBones));
        humanBodyBoneTransforms = new Transform[humanBodyBones.Length];
        humanBodyBoneRotations = new Quaternion[humanBodyBones.Length];
    }
    
    public void SetProxyValues(Dictionary<BlendShapeKey, float> dict) {
        foreach (var kv in dict) {
            proxy.AccumulateValue(kv.Key, kv.Value, 1f);
        }
    }
    
    public void SetExternalPerfectSync(Dictionary<string, float> blendshapes, float time) {
        if (blendshapes == null) {
            proxy.ClearPerfectSync();
            return;
        }
        foreach (var entry in blendshapes)
            proxy.SetPerfectSync(entry.Key, entry.Value, time, 1f);
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
        proxy.SetWeight(blendShapeWeight);
        proxy.Clear();
        FindFaceMesh();
        if (!skipApply)
            UpdateExpression();
        if (!skipApply)
            BlinkEyes();
        bool doMouthTracking = true;
        #if WINDOWS_BUILD
        ReadAudio();
        if (lipSync) {
            if (!skipApply)
                doMouthTracking = ApplyVisemes();
        }
        #endif
        if (doMouthTracking && !skipApply)
            ApplyMouthShape();
        if (!skipApply)
            UpdatePerfectSync();
        if (!skipApply && vrmBlendShapeProxy != null && browClips == null)
            proxy.Apply();
        if (!skipApply)
            UpdateBrows();
        if (!skipApply && vrmBlendShapeProxy != null && browClips != null)
            proxy.Apply();
    }
    
    void LateUpdate() {
        if (!skipApplyJaw && jawBone != null && jawBoneAnimation != null && !haveJawParameter) {
            jawBone.localRotation = jawRotation;
        }
        if (!skipApplyEyes && gazeTracking && !only30Points) {
            UpdateGaze();
        }
        // If blendshapes are received over VMC, apply expressions afterwards if they should be applied
        if (skipApply && stillApplyExpressions) {
            UpdateExpression();
        }
        proxy.Apply();
    }
    
    void FixedUpdate() {
        if (openSeeIKTarget.fixedUpdate) {
            RunUpdates();
        }
    }

    void Update() {
        fakeTimeTime = Time.time;
        if (inited && lastSmoothing != smoothAmount) {
            #if WINDOWS_BUILD
            OVRLipSync.SendSignal(context, OVRLipSync.Signals.VisemeSmoothing, smoothAmount, 0);
            #endif
            lastSmoothing = smoothAmount;
        }
        if (!openSeeIKTarget.fixedUpdate) {
            RunUpdates();
        }
    }

    void DestroyContext() {
        #if WINDOWS_BUILD
        active = false;
        lock (this) {
            if (context != 0) {
                if (OVRLipSync.DestroyContext(context) != OVRLipSync.Result.Success) {
                    Debug.LogError("OVRLipSyncContextBase.OnDestroy ERROR: Could not delete Phoneme context.");
                }
                context = 0;
            }
        }
        #endif
    }

    void OnDestroy() {
        DestroyContext();
    }
    
    public void SetAnimateJawBone(bool value) {
        animateJawBone = value;
    }
    
    public void SetSmartWinkThreshold(float v) {
        smartWinkThreshold = v;
    }

    public void SetGazeSmoothing(float v) {
        gazeSmoothing = v;
    }

    public void SetVisemeNormalization(bool v) {
        visemeNormalization = v;
    }

}