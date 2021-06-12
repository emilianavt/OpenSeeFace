using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace OpenSee {
    public class InterpolatedMap<T, V, U> where V : InterpolatedValue<U>, new() {
        private Dictionary<T, V> map = new Dictionary<T, V>();
        private float clamp = 1f;
        private float smoothing = 0f;
        private bool gotDefault = false;
        private U defaultValue;

        public void Clear() {
            map.Clear();
        }
        
        public void SetSmoothing(float smoothing) {
            this.smoothing = smoothing;
            foreach (var entry in map) {
                if (entry.Value != null)
                    entry.Value.SetSmoothing(smoothing);
            }
        }

        public void SetClamp(float clamp) {
            this.clamp = clamp;
            foreach (var entry in map) {
                if (entry.Value != null)
                    entry.Value.SetClamp(clamp);
            }
        }
        
        void Prepare(T key) {
            if (!map.ContainsKey(key)) {
                map.Add(key, new V());
                map[key].SetSmoothing(smoothing);
                map[key].SetClamp(clamp);
            }
        }
        
        public void SetValue(T key, U v) {
            Prepare(key);
            map[key].SetValue(v);
        }
        
        public void Store(T key, U v) {
            Prepare(key);
            map[key].UpdateValue(v);
        }

        public void Store(T key, U v, double nowT) {
            if (!map.ContainsKey(key)) {
                map.Add(key, new V());
                map[key].SetSmoothing(smoothing);
                map[key].SetClamp(clamp);
            }
            map[key].UpdateValue(v, nowT);
        }
        
        public U GetDefault() {
            if (gotDefault)
                return defaultValue;
            V tmp = new V();
            defaultValue = tmp.GetDefault();
            return defaultValue;
        }
        
        public bool Check(T key) {
            return map.ContainsKey(key);
        }
        
        public U Get(T key) {
            if (!map.ContainsKey(key))
                return GetDefault();
            return map[key].Interpolate();
        }
    }

    abstract public class InterpolatedValue<V> {
        private float interpolateT = 0f;
        private float updateT = 0f;
        private double lastT = 0f;
        private double currentT = 0f;
        private float clamp = 1f;
        private float smoothing = 0f;
        private int gotData = 0;

        private bool haveDefault = false;
        private V defaultValue;

        private bool valueSet = false;
        private V dataValue;
        private V dataValueLast;

        abstract public V ValueLerpUnclamped(V a, V b, float t);
        abstract public V GetBuiltinDefault();
        
        public void SetDefault(V defaultValue) {
            this.defaultValue = defaultValue;
            haveDefault = true;
        }
        
        public void ResetDefault() {
            haveDefault = false;
        }
        
        public V GetDefault() {
            if (haveDefault)
                return defaultValue;
            return GetBuiltinDefault();
        }
        
        public void SetSmoothing(float smoothing) {
            this.smoothing = smoothing;
        }

        public void SetClamp(float clamp) {
            this.clamp = clamp;
        }
        
        public void SetValue(V v) {
            dataValueLast = v;
            dataValue = v;
            valueSet = true;
        }
        
        public InterpolatedValue(float smoothing = 0f, float clamp = 1f) {
            this.smoothing = smoothing;
            this.clamp = clamp;
        }
        
        public void UpdateValue(V v) {
            UpdateValue(v, Time.realtimeSinceStartup);
        }
        
        public void UpdateValue(V v, double nowT) {
            if (nowT < 0f)
                return;
            lock (this) {
                if (nowT < currentT || gotData == 0) {
                    // Reinitialize
                    gotData = 1;
                    updateT = Time.time;
                    lastT = 0f;
                    currentT = nowT;
                    dataValueLast = GetDefault();
                    dataValue = ValueLerpUnclamped(v, dataValueLast, smoothing);
                } else if (nowT == currentT) {
                    // Change value
                    dataValue = ValueLerpUnclamped(v, dataValueLast, smoothing);
                } else {
                    // Regular operation
                    updateT = Time.time;
                    lastT = currentT;
                    currentT = nowT;
                    if (gotData < 2)
                        gotData++;
                    dataValueLast = dataValue;
                    dataValue = ValueLerpUnclamped(v, dataValueLast, smoothing);
                }
            }
        }
        
        public V Interpolate() {
            lock (this) {
                interpolateT = Time.time;
                if (gotData < 2) {
                    if (valueSet)
                        return dataValue;
                    else
                        return GetDefault();
                }
                float t = Mathf.Min((interpolateT - updateT) / (float)(currentT - lastT), clamp);
                return ValueLerpUnclamped(dataValueLast, dataValue, t);
            }
        }
    }
    
    public class InterpolatedColor : InterpolatedValue<Color> {
        public override Color ValueLerpUnclamped(Color a, Color b, float t) {
            return Color.LerpUnclamped(a, b, t);
        }
        public override Color GetBuiltinDefault() {
            return Color.clear;
        }
    }
    
    public class InterpolatedQuaternion : InterpolatedValue<Quaternion> {
        public override Quaternion ValueLerpUnclamped(Quaternion a, Quaternion b, float t) {
            return Quaternion.SlerpUnclamped(a, b, t);
        }
        public override Quaternion GetBuiltinDefault() {
            return Quaternion.identity;
        }
    }

    public class InterpolatedVector2 : InterpolatedValue<Vector2> {
        public override Vector2 ValueLerpUnclamped(Vector2 a, Vector2 b, float t) {
            return Vector2.LerpUnclamped(a, b, t);
        }
        public override Vector2 GetBuiltinDefault() {
            return Vector2.zero;
        }
    }

    public class InterpolatedVector3 : InterpolatedValue<Vector3> {
        public override Vector3 ValueLerpUnclamped(Vector3 a, Vector3 b, float t) {
            return Vector3.LerpUnclamped(a, b, t);
        }
        public override Vector3 GetBuiltinDefault() {
            return Vector3.zero;
        }
    }

    public class InterpolatedVector4 : InterpolatedValue<Vector4> {
        public override Vector4 ValueLerpUnclamped(Vector4 a, Vector4 b, float t) {
            return Vector4.LerpUnclamped(a, b, t);
        }
        public override Vector4 GetBuiltinDefault() {
            return Vector4.zero;
        }
    }

    public class InterpolatedFloat : InterpolatedValue<float> {
        public override float ValueLerpUnclamped(float a, float b, float t) {
            return Mathf.LerpUnclamped(a, b, t);
        }
        public override float GetBuiltinDefault() {
            return 0f;
        }
    }
}
