using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class Check_Collision : MonoBehaviour
{

    public Agent8 agent_script;
    public GameObject box;

    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.tag == "Box")
        {
            agent_script.check_collide = true;//Debug.Log("Box collide with something");
            //Debug.Log("Can't Stack");
            Debug.Log("Collide");
        }
    }

    void OnCollisionStay(Collision other)
    {
        if (other.gameObject.tag == "Box")
        {
            agent_script.check_collide = true;//Debug.Log("Box collide with something");
            //Debug.Log("Can't Stack");
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        box = this.gameObject;
    }

    // Update is called once per frame
    void Update()
    {


    }
}
