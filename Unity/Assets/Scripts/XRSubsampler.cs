using System.Collections;
using System.Collections.Generic;
using UnityEngine.XR;
using UnityEngine;

//Lets you set rendering resolution for VR.
//Use this in combination with relief mapping quality and rendertexture resolution to deal with performance.
public class XRSubsampler : MonoBehaviour {
    public float resolutionScale = 0.4f;

    void Start() {
        XRSettings.eyeTextureResolutionScale = resolutionScale;
    }
}
