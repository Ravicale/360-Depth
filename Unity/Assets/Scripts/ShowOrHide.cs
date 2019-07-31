using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShowOrHide : MonoBehaviour {
    public bool beginVisible = false;
    private Renderer rend;

    void Start() {
        rend = GetComponent<Renderer>();
        rend.enabled = beginVisible;
    }

    // Toggle the Object's visibility each second.
    void Update() {
        if (Input.GetKeyUp(KeyCode.Space)) {
            rend.enabled = !rend.enabled;
        }
    }
}
