using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class Jumper : Agent
{
    public Rigidbody rBody;
    public Transform Switch;
    public GameObject Mover;
    public GameObject RedWall;
    public float forceMultiplier = 10;
    public bool switch_on = false;
    public override void Initialize()
    {
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
        this.transform.localPosition = new Vector3(-2.5f, 0.5f, -2.5f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions
        sensor.AddObservation(Switch.localPosition.y);
        sensor.AddObservation(this.transform.localPosition.y);
        sensor.AddObservation(Mover.transform.localPosition);
    }

    void OnCollisionStay(Collision other)
    {
        if(other.gameObject.tag == "Switch")
        {
            switch_on = true;
        }
        else
        {
            switch_on = false;
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.y = actionBuffers.ContinuousActions[0];
        Debug.Log(controlSignal.y);
        rBody.AddForce(controlSignal * forceMultiplier);

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Switch.localPosition);

        // Reached target
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        if (Input.GetKey(KeyCode.Space))
            continuousActionsOut[0] = 1f;
        if (Input.GetKey(KeyCode.LeftControl))
            continuousActionsOut[0] = -1f;
    }
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (switch_on == true)
            RedWall.SetActive(false);
        else
            RedWall.SetActive(true);
    }
}
