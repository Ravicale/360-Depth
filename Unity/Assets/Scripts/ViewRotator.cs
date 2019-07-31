using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Moves camera rig around to emulate VR.
public class ViewRotator : MonoBehaviour {
    public float turnSpeed = 1.0f; //Rate at which camera rotates.
    public float moveSpeed = 0.02f;
    private Transform pos; //Transform object.
    private Vector3 initialPos;
    public Vector3 currOffset;
    private Panorama panorama;
    private float turnX = 0.0f; //Direction to spin.
    private float turnY = 0.0f;
    private enum direct {left = -1, front = -1, none = 0, right = 1, back = 1};
    private direct xOffset = direct.none;
    private direct yOffset = direct.none;
    private direct xMove = direct.none;
    private direct yMove = direct.none;

    private float rotX = 0.0f; //Current rotation values.
    private float rotY = 0.0f;

    void Start() {
        pos = GetComponent<Transform>();
        initialPos = pos.position;
        currOffset = new Vector3(0.0f, 0.0f, 0.0f);
        panorama = GameObject.Find("Panorama").GetComponent<Panorama>();
    }

    //Getting user input.
    void Update() {
        turnY = Input.GetAxis("Horizontal");
        turnX = Input.GetAxis("Vertical");

        if (Input.GetKey(KeyCode.J)) {
            xOffset = direct.left;
        } else if (Input.GetKey(KeyCode.L)) {
            xOffset = direct.right;
        } else {
            xOffset = direct.none;
        }

        if (Input.GetKey(KeyCode.I)) {
            yOffset = direct.front;
        } else if (Input.GetKey(KeyCode.K)) {
            yOffset = direct.back;
        } else {
            yOffset = direct.none;
        }

        if (Input.GetKey(KeyCode.F)) {
            xMove = direct.left;
        }
        else if (Input.GetKey(KeyCode.H)) {
            xMove = direct.right;
        } else {
            xMove = direct.none;
        }

        if (Input.GetKey(KeyCode.T)) {
            yMove = direct.back;
        } else if (Input.GetKey(KeyCode.G)) {
            yMove = direct.front;
        } else {
            yMove = direct.none;
        }
    }

    //Performing transformations.
    void FixedUpdate() {
        //Update internal rotation values.
        rotX = rotX + turnSpeed * turnX;
        rotY = rotY + turnSpeed * turnY;

        //Prevent gimbal lock on looking up/down.
        if (rotX > 89.0f) {
            rotX = 89.0f;
        } else if (rotX < -89.0f) {
            rotX = -89.0f;
        }
        
        //Apply new rotation values to transformation.
        pos.eulerAngles = new Vector3(rotX, rotY, 0.0f);

        //Handle faked camera motion for shader.
        transform.Translate(new Vector3((float)xOffset * 0.02f, 0.0f, (float)yOffset * 0.02f), Space.Self);
        currOffset += pos.position - initialPos; //Get coordinates as if camera was in new position.
        panorama.SetCameraOffset(currOffset);
        pos.position = initialPos;

        //Physical camera movement.
        transform.Translate(new Vector3((float)xMove * 0.02f, 0.0f, (float)yMove * 0.02f), Space.Self);
        initialPos = pos.position;
    }
}
