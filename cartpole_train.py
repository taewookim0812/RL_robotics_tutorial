# -*- coding: utf-8 -*-
"""
CartPole simple state version
===============================================================
This code is based on the official pytorch tutorial DQN code.
"https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"

**Author** modified by Taewoo Kim <https://github.com/go-blin>

**changes**
The original code takes input state as image sequences,
but in this code simple state of CartPole such as
cart position, pole angle and its velocity is used
as state. And some minor parts are modified.

===============================================================
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from ModelDescriptor import cartpole_DQN
import os


env = gym.make('CartPole-v1').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env.reset()


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_PERIOD = 200
EPS_DECAY = -EPS_PERIOD / math.log(EPS_END)

TARGET_UPDATE = 10
LOG_INTERVAL = 20
MODEL_SAVE_PERIOD = 10

policy_net = cartpole_DQN().to(device)
target_net = cartpole_DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)
memory = ReplayMemory(10000)


steps_done = 0


def get_state(state):
    state = np.ascontiguousarray(state, dtype=np.float64)
    state = torch.from_numpy(state).unsqueeze(0).type(
        torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    return state

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('CartPole Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


episode_rewards = []

def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('CartPole Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(rewards_t) >= 100:
    #     means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()


save_dir = './model/CartPole'
def save_model(num_episodes):
    # save model parameters
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saveName = 'CartPole' + '_epi' + str(num_episodes) + '.pt'
    torch.save(target_net.state_dict(), save_dir + '/' + saveName)


total_num_steps = 0
num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    current_state = env.reset()
    current_state = get_state(current_state)

    rewards_step = 0
    for t in count():

        # Select and perform an action
        action = select_action(current_state)
        state, reward, done, _ = env.step(action.item())
        state = get_state(state)

        rewards_step += reward

        # TODO, CartPole visualization
        # env.render()

        # Observe new state
        if not done:
            next_state = state
        else:
            next_state = None
            # reward -= 10.0
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(current_state, action, next_state, reward)

        # Move to the next state
        current_state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()  # TODO
            break

    total_num_steps += t
    episode_rewards.append(rewards_step)
    plot_rewards()

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save model
    if i_episode % MODEL_SAVE_PERIOD == 0:
        # print('Model was saved...')
        save_model(i_episode)

    # Log interval
    if i_episode % LOG_INTERVAL == 0:
        mean = np.mean(episode_durations[-100:])
        print('Updates {}, num steps {}, mean duration {:.1f}'.
              format(i_episode, total_num_steps, mean))

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()