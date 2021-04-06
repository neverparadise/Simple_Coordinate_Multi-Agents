import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ActorCritic import ActorCritic

# 1. 환경을 가져오기
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()

# 2. 환경에 정의된 에이전트의 행동 정보 가져오기
mover_behavior = list(env.behavior_specs)[0]
jumper_behavior = list(env.behavior_specs)[1]
print(f"Name of the behavior : {mover_behavior}")
print(f"Name of the behavior : {jumper_behavior}")
mover_spec = env.behavior_specs[mover_behavior]
jumper_spec = env.behavior_specs[jumper_behavior]

# 3. 에이전트의 observation 정보를 가져오기
print("Number of observations : ", (mover_spec.observation_specs[0].shape))
print("Number of observations : ", (jumper_spec.observation_specs[0].shape))

mover_obs_size = mover_spec.observation_specs[0].shape[0]
jumper_obs_size = jumper_spec.observation_specs[0].shape[0]
print("mover_obs_size : {} ".format(mover_obs_size))
print("jumper_obs_size : {} ".format(jumper_obs_size))

# 4. 에이전트의 action 정보를 가져오기
mover_action_size = mover_spec.action_spec.continuous_size
jumper_action_size = jumper_spec.action_spec.continuous_size
print(mover_action_size)
print(jumper_action_size)

# 5. 네트워크 두 개 정의
mover_network = ActorCritic(mover_obs_size, mover_action_size)
jumper_network = ActorCritic(jumper_obs_size, jumper_action_size)

env.reset()

for n_epi in range(10000):
    done = False
    #env.reset()을 하면 mlagents에서는 어떤 함수가 호출되는가?
    # EpisodeBegin()? Initialize()? 둘 중 하나 일 것이다.
    m_decision_steps, m_terminal_steps = env.get_steps(mover_behavior)
    j_decision_steps, j_terminal_steps = env.get_steps(jumper_behavior)
    episode_rewards = 0
    while not done:
        for t in range(10):
            m_agent = m_decision_steps.agent_id[0]
            j_agent = j_decision_steps.agent_id[0]

            m_obs = np.array(m_decision_steps.obs)
            j_obs = np.array(j_decision_steps.obs)

            print(m_obs.squeeze(0).shape)
            print(type(m_obs))
            print(type(j_obs))

            # action sampling
            #mover_action_prob = mover_network.pi(torch.from_numpy(m_obs))
            #jumper_action_prob = jumper_network.pi(torch.from_numpy(j_obs))
            #m = Categorical(mover_action_prob)
            #j = Categorical(jumper_action_prob)
            #m_a = m.sample().numpy()
            #j_a = j.sample().numpy()

            # 신경망의 출력을 변수할당한다. 변수는 torch.tensor
            m_a = mover_network.pi(torch.from_numpy(m_obs))
            j_a = jumper_network.pi(torch.from_numpy(j_obs))

            # [1, 2, 1] -> [2, 1]
            m_a = m_a.squeeze(0)
            j_a = j_a.squeeze(0)
            print(m_a)
            print(j_a)

            # numpy로 변환 후, 액션튜플에 넣어준다.

            m = ActionTuple(continuous=m_a.detach().numpy())
            j = ActionTuple(continuous=j_a.detach().numpy())


            env.set_actions(mover_behavior, m)
            env.set_actions(jumper_behavior, j)

            # Go to next step
            env.step()

            m_decision_steps, m_terminal_steps = env.get_steps(mover_behavior)
            j_decision_steps, j_terminal_steps = env.get_steps(jumper_behavior)

            m_nobs = m_decision_steps.obs
            j_nobs = j_decision_steps.obs

            mover_network.put_data((m_obs, m_a, m_decision_steps.reward, m_nobs, done))
            jumper_network.put_data((j_obs, j_a, j_decision_steps.reward, j_nobs, done))

            if(m_agent in m_decision_steps):
                episode_rewards += m_decision_steps[m_agent].reward
                episode_rewards += j_decision_steps.reward

            if (m_agent in m_terminal_steps):
                episode_rewards += m_terminal_steps[m_agent].reward
                done = True

            if done:
                break
        mover_network.train_net()
        jumper_network.train_net()

    print(episode_rewards)

env.close()