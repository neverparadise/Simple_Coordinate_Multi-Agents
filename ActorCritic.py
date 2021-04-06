import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

gamma = 0.98
learning_rate = 0.001

class ActorCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super(ActorCritic, self).__init__()
        # 레이어를 두 개로 나눌 수도 있음.
        # 레이어를 나누는게 일반적인 솔루션

        # 다변수 분포를 만들어서 할 수도 있음.

        self.fc1 = nn.Linear(obs_size, 256)
        self.fc_pi = nn.Linear(256, action_size)
        self.fc_pi2 = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256, 1)
        self.data = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        #prob = F.softmax(x, dim=softmax_dim)
        act = F.tanh(x)
        return act

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        #a_lst = [[-0.5, 1.0], [-0.2, 0.1]]
        #a_lst = [[0.5], [-0.1], ..]
        #print(a_lst)
        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = a_lst
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_lst, dtype=torch.float)
        self.data = []

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        # a[0] = x좌표 움직임
        # a[1] = z좌표 움직임
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=-1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()