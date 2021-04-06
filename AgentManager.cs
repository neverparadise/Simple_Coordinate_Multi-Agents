using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class AgentManager : MonoBehaviour
{
    // Start is called before the first frame update
    public Mover mover_script;
    public GameObject mover;
    public Jumper jumper_script;
    public GameObject jumper;
    public Transform Target;
    public Transform Switch;

    public void InitializeAgents()
    {
        mover_script.Initialize();
        jumper_script.Initialize();
    }

    public void SetAgentsRewards(float reward)
    {
        mover_script.SetReward(reward);
        jumper_script.SetReward(reward);
    }

    public void EndAgentsEpisode()
    {
        mover_script.EndEpisode();
        jumper_script.EndEpisode();
    }
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // Rewards
        float distanceToTarget = Vector3.Distance(mover.transform.localPosition, Target.localPosition);
        float distanceToTarget2 = Vector3.Distance(jumper.transform.localPosition, Switch.localPosition);
        //Debug.Log(distanceToTarget);
        // Reached target
        if (distanceToTarget2 < 1.0f)
        {
            Debug.Log("Switch On");
            jumper_script.SetReward(0.5f);
        }

        if (distanceToTarget < 1.4f)
        {
            SetAgentsRewards(1.0f);
            EndAgentsEpisode();
            InitializeAgents();
        }
        // Fell off platform
        else if (mover.transform.localPosition.y < 0)
        {
            Debug.Log("Mover fell down...");
            SetAgentsRewards(-1.0f);
            EndAgentsEpisode();
            InitializeAgents();
        }
    }
}
