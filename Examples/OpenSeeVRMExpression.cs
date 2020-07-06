using System.Collections.Generic;
using UnityEngine;
using VRM;

namespace OpenSee {

public class OpenSeeVRMExpression : MonoBehaviour
{
    public OpenSee openSee;
    public OpenSeeExpression openSeeExpression;
    public VRMBlendShapeProxy vrmBlendShapeProxy;
    public string expressionNeutral = "neutral";
    public string expressionFun = "smile";
    public string expressionSorrow = "sad";
    public string expressionAngry = "angry";
    public string expressionJoy = "shocked";
    
    private BlendShapeKey lastKey;
    
    void Start()
    {
    }
    
    void Update()
    {
        if (openSeeExpression == null || vrmBlendShapeProxy == null) {
            return;
        }
        float rightEyeOpen = 1f;
        float leftEyeOpen = 1f;
        if (openSee != null) {
            var openSeeData = openSee.trackingData;
            if (openSeeData == null || openSeeData.Length < 1)
                return;
            int idx = -1;
            int i = 0;
            foreach (var item in openSeeData) {
                if (item.id == openSeeExpression.faceId)
                    idx = i;
                i++;
            }
            if (idx > -1) {
                rightEyeOpen = openSeeData[idx].rightEyeOpen;
                leftEyeOpen = openSeeData[idx].leftEyeOpen;
            }
        }
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Neutral), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Fun), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Sorrow), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Angry), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Joy), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink_R), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Blink_L), 0f);
        if (openSeeExpression.expression == expressionFun) {
            vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Fun), 1f);
            return;
        } else if (openSeeExpression.expression == expressionSorrow) {
            vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Sorrow), 1f);
            return;
        } else if (openSeeExpression.expression == expressionAngry) {
            vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Angry), 1f);
            return;
        } else if (openSeeExpression.expression == expressionJoy) {
            vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Joy), 1f);
            return;
        } else {
            /*if (rightEyeOpen < 0.05) {
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_R), 1f);
            }
            if (leftEyeOpen < 0.05) {
                vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_L), 1f);
            }*/
        }
        vrmBlendShapeProxy.ImmediatelySetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Neutral), 1f);
    }
}

}