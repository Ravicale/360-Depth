using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Slideshow : MonoBehaviour {

    public Material[] Images;

    void Update() {
        if (Input.GetKey(KeyCode.Alpha1)) {
            GetComponent<Renderer>().material = Images[0];
        }
        if (Input.GetKey(KeyCode.Alpha2)) {
            GetComponent<Renderer>().material = Images[1];
        }
        if (Input.GetKey(KeyCode.Alpha3)) {
            GetComponent<Renderer>().material = Images[2];
        }
        if (Input.GetKey(KeyCode.Alpha4)) {
            GetComponent<Renderer>().material = Images[3];
        }
        if (Input.GetKey(KeyCode.Alpha5)) {
            GetComponent<Renderer>().material = Images[4];
        }
        if (Input.GetKey(KeyCode.Alpha6)) {
            GetComponent<Renderer>().material = Images[5];
        }
        if (Input.GetKey(KeyCode.Alpha7)) {
            GetComponent<Renderer>().material = Images[6];
        }
        if (Input.GetKey(KeyCode.Alpha8)) {
            GetComponent<Renderer>().material = Images[7];
        }
        if (Input.GetKey(KeyCode.Alpha9)) {
            GetComponent<Renderer>().material = Images[8];
        }
    }
}
