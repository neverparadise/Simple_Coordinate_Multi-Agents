
import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment

# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
# Start interacting with the environment.

# 1. To reset the environment
env.reset()

# 2. Get the Behavior specification from the environment
# We will only consider the first Behavior
behavior_name1 = list(env.behavior_specs)[0]
behavior_name2 = list(env.behavior_specs)[1]
print(f"Name of the behavior : {behavior_name1}")
spec1 = env.behavior_specs[behavior_name1]
spec2 = env.behavior_specs[behavior_name2]

print(spec1)

# Examine the number of observations per Agent
# How many actions are possible ?

print("Number of observations : ", (spec1.observation_specs[0]))
if spec1.action_spec.continuous_size > 0:
  print(f"There are {spec1.action_spec.continuous_size} continuous actions")
if spec1.action_spec.is_discrete():
  print(f"There are {spec1.action_spec.discrete_size} discrete actions")

# 3. Get the steps from the Environment
decision_steps1, terminal_steps1 = env.get_steps(behavior_name1)
decision_steps2, terminal_steps2 = env.get_steps(behavior_name2)
print(decision_steps1.agent_id)
print(decision_steps2.agent_id)

# 4. Set actions for each behavior
env.set_actions(behavior_name1, spec1.action_spec.empty_action(len(decision_steps1)))
env.set_actions(behavior_name2, spec2.action_spec.empty_action(len(decision_steps2)))

# 5. Move the simulation forward
# env.step()

# 6. Run the Environment for a few episodes
for episode in range(10):
    env.reset()   # Call the EpisodeBegin() Function

    # 별도의 에이전트가 존재한다는 것은 각각의 의사결정 단계와 종료단계가 존재해야하는것인가?
    # 사실 우리가 원하는 것은 하나의 에피소드 시작, 종료 조건을 가진 환경에서 여러개의 에이전트가 각각 행동하는 것을 원하는 것이다.
    # 그렇다면 behavior는 지금 파이썬 API 구조상 하나밖에 존재할 수 없는 것 아닌가?
    # 객체지향을 끝판왕으로 활용해서 리셋(시작, 종료)를 해결해야함
    # done1이랑 done2를 만들어서 종료조건을 따로 관리해야할 것 같다.
    # adversarial cooperative가 같이 있는 multi object multi agent 환경 구현하면 엄청 공부가 많이 될듯

    # 이슈 1 리셋
    # 이슈 2
    decision_steps1, terminal_steps1 = env.get_steps(behavior_name1)
    decision_steps2, terminal_steps2 = env.get_steps(behavior_name2)
    print("terminal steps: ", terminal_steps1.obs)
    print("terminal steps: ", terminal_steps2.values())
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False
    episode_rewards = 0 # For the tracked agent
    while not done:
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]
        if tracked_agent == -1 and len(decision_steps1) >= 1:
            tracked_agent = decision_steps1.agent_id[0]
    
        # Generate an action for all agents
        action1 = spec1.action_spec.random_action(len(decision_steps1))
        print("action1: ", action1.continuous)
        action2 = spec2.action_spec.random_action(len(decision_steps2))

        print(action2)
        # Set the actions
        env.set_actions(behavior_name1, action1)
        env.set_actions(behavior_name2, action2)
        env.step()

        ################ 여기서 의문점!!! 각 action을 줬으면 에이전트 C# 스크립트에 정의를 해주어야 할텐데 Episode Begin을 한쪽에만 정의해도 괜찮은가?
        # 여러분 도와주세요!


        #env.set_action_for_agent('Behavior1?team=0', 1, action1)
        #env.set_action_for_agent('Behavior2?team=0', 0, action2)

        # Move the simulation forward
#        env.step()



        # Get the new simulation results
        decision_steps1, terminal_steps1 = env.get_steps(behavior_name1)
        decision_steps2, terminal_steps2 = env.get_steps(behavior_name2)
        if tracked_agent in decision_steps1:  # The agent requested a decision
            episode_rewards += decision_steps1[tracked_agent].reward
        if tracked_agent in terminal_steps1:  # The agent terminated its episode
            episode_rewards += terminal_steps1[tracked_agent].reward
            done = True
        print(f"Total rewards for episode {episode} is {episode_rewards}")

env.close()
print("Closed environment")


#behavior_names = list(env.get_behavior_names())
# specs = env.behavior_specs #spces는 딕셔너리임
# specs_name_list = list(specs.keys())
# print("spec_name_list : {}".format(specs_name_list))
# num_agents = len(specs_name_list)
# print("num agents : ", num_agents)
#
# decision_steps, terminal_steps = env.get_steps(specs_name_list[0])
# decision_steps2, terminal_steps2 = env.get_steps(specs_name_list[1])
#
# print("Behavior Spec")
# behavior_spec0 = specs[specs_name_list[0]]
# print(behavior_spec0)
# print(type(behavior_spec0))
# print(behavior_spec0.action_spec)
#
# action_size0 = behavior_spec0.action_spec.continuous_size
# print(action_size0)
#
#
#
# for episode in range(3):
#     tracked_agent = -1
#     done = False
#     episode_rewards = 0
#     while not done:
#         if(tracked_agent == -1 and len(num_agents) == 1):
#             tracked_agent = decision_steps.agent_id[0]
#
#         action = np.random.randint(num_agents, size=(num_agents, specs.items(action_size0)))