using System.Collections.Generic;
using UnityEngine;
using VRM;

namespace OpenSee {

public class OpenSeeVRMExpression : MonoBehaviour
{
    public OpenSeeExpression openSeeExpression;
    public VRMBlendShapeProxy vrmBlendShapeProxy;
    public string expressionNeutral = "neutral";
    public string expressionFun = "smile";
    public string expressionSorrow = "sad";
    public string expressionAngry = "angry";
    public string expressionJoy = "shocked";
    
    private BlendShapeKey lastKey;
    private string lastExpression = "";
    
    void Start()
    {
    }
    
    void Update()
    {
        if (openSeeExpression == null || vrmBlendShapeProxy == null) {
            return;
        }
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Neutral), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Fun), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Sorrow), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Angry), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Joy), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_R), 0f);
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Blink_L), 0f);
        if (openSeeExpression.expression == expressionFun) {
            vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Fun), 1f);
            return;
        } else if (openSeeExpression.expression == expressionSorrow) {
            vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Sorrow), 1f);
            return;
        } else if (openSeeExpression.expression == expressionAngry) {
            vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Angry), 1f);
            return;
        } else if (openSeeExpression.expression == expressionJoy) {
            vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Joy), 1f);
            return;
        }
        vrmBlendShapeProxy.ImmediatelySetValue(new BlendShapeKey(BlendShapePreset.Neutral), 1f);
    }
}

}