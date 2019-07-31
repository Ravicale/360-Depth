using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Panorama : MonoBehaviour {
    private Material mat;

    //Manages camera offset value in shader (fake camera position), used for debugging.
    void Start() {
        mat = GetComponent<Renderer>().material;
        mat.SetVector("_CameraOffset", new Vector3(0.0f, 0.0f, 0.0f));
    }

    public void SetCameraOffset(Vector3 pos) {
        mat.SetVector("_CameraOffset", pos);
    }
}
