"""
CartPole evaluation
========================================
**Author** Taewoo Kim

"""

import gym
import torch
import numpy as np
import os

from ModelDescriptor import cartpole_DQN

env = gym.make('CartPole-v1')

load_dir = './model'
num_epi = 100
env_name = 'CartPole_epi' + str(num_epi) + '.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = cartpole_DQN().to(device)

policy_net.load_state_dict(torch.load(os.path.join(load_dir + '/' + env_name)))


def get_state(state):
    state = np.ascontiguousarray(state, dtype=np.float64)
    state = torch.from_numpy(state).unsqueeze(0).type(
        torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    return state

def eval_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

state = env.reset()
state = get_state(state)
nIter = 10000
for t in range(nIter):
    action = eval_action(state)
    state, reward, done, _ = env.step(action.item())
    state = get_state(state)
    env.render()

    if done:
        state = env.reset()
        state = get_state(state)


print('Evaluation Complete!')