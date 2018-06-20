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
from torch.autograd import Variable
import os


# Custom library
from ModelDescriptor import reacher_DQN
from DataPlotter import *


env = gym.make('Reacher-v2').unwrapped


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()


if device == torch.device("cuda"):
    use_cuda = True
    print('cuda is available')
else:
    use_cuda = False
    print('cuda is NOT available')

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


action_range = [0.05, 0.01, 0.00, -0.01, -0.05]
action_list = []
for j in action_range:
    temp = []
    for i in action_range:
        temp.append(j)
        temp.append(i)
        action_list.append(temp)
        temp = []

print(action_list)



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


# user-defined state
# # [q1, q2, q1_dot, q2_dot, fx, fy, fz, tx, ty, tz]
# def get_state():
#     theta = env.sim.data.qpos.flat[:2]
#     theta_dot = env.sim.data.qvel.flat[:2]
#     finger = env.get_body_com("fingertip")
#     target = env.get_body_com("target")
#     state = np.concatenate((theta, theta_dot, finger, target))
#     state = np.ascontiguousarray(state, dtype=np.float32)
#     return torch.from_numpy(state).unsqueeze(0).type(Tensor)


def get_state(state):
    state = np.ascontiguousarray(state, dtype=np.float64)
    state = torch.from_numpy(state).unsqueeze(0).type(
        torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    return state



BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_PERIOD = 150000
EPS_DECAY = -EPS_PERIOD/math.log(EPS_END)
TARGET_UPDATE = 200
LOG_INTERVAL = 100

RHO = 1/2 - (1-EPS_START)   # amplitude for cos
DELTA = 0.01


policy_net = reacher_DQN().to(device)
target_net = reacher_DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# policy_net.train()

print('policy net: ', policy_net)

if use_cuda:
    policy_net.cuda()
    target_net.cuda()


optimizer = optim.RMSprop(policy_net.parameters(), lr=2e-4)
memory = ReplayMemory(1000000)

steps_done = 0

# returns the index of possible actions
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done/EPS_DECAY)

    # eps_threshold = math.exp(-1. * steps_done / EPS_DECAY) * \
    #                 (RHO * math.cos(2*math.pi*DELTA*steps_done) + 1/2)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            result = policy_net(state)
        step_Q.append(result.data.max(1)[0])
        return result.data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(len(action_list))]])


# from action index to joint velocity
def action_interpreter(action_index):
    return np.array(action_list[int(action_index[0])])



savePath = './model/Reacher'
def savePlots():
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    with open(savePath + '/Average_duration_per_Epi.txt', 'w') as f:
        for s in episode_durations:
            f.write(str(s) + '\n')

    with open(savePath + '/Average_reward_per_Epi.txt', 'w') as f:
        for s in episode_rewards:
            f.write(str(s) + '\n')

    with open(savePath + '/Average_Q_per_Epi.txt', 'w') as f:
        for s in episode_Q:
            f.write(str(s) + '\n')

    with open(savePath + '/Average_Err_per_Epi.txt', 'w') as f:
        for s in episode_Err:
            f.write(str(s) + '\n')


def save_model(num_episodes):
    # save model parameters
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    saveName = 'ReacherModel' + '_epi' + str(num_episodes) + '.pt'
    torch.save(target_net.state_dict(), savePath + '/' + saveName)


def optimize_model():
    if len(memory) < BATCH_SIZE * 10000:
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



REQ_START = 0.1
REQ_END = 0.005
REQ_PERIOD = 150000
REQ_DECAY = -REQ_PERIOD/math.log(REQ_END)

MODEL_SAVE_PERIOD = 1000
requirement = 0.1
total_num_steps = 1
num_episodes = 2000000
num_steps_per_epi = 50

for i_episode in range(num_episodes):
    # Initialize the environment and state
    current_state = env.reset()
    current_state = get_state(current_state)

    # To see the progress of learning
    epi_rewards = 0
    step_Q = []
    step_errs = []
    for t in range(num_steps_per_epi):
        # Select and perform an action
        action = select_action(current_state)
        jointVel = action_interpreter(action)  # + np.random.normal(0.01, 0.02)  # action noise for exploration

        state, reward, done, info = env.step(jointVel)   # desired joint velocity
        state = get_state(state)

        # TODO, visualization of Reacher
        # env.render()

        err_dist = np.abs(info['reward_dist'])
        err_ctrl = np.abs(info['reward_ctrl'])
        # print('err_dist: ', err_dist, '  req: ', requirement, '  t: ', t)
        step_errs.append(err_dist)


        # if err_dist < requirement and err_ctrl < 0.05:    # TODO, target distance and velocity
        #     done = True
        #     reward += 0.1
        #
        # requirement = REQ_END + (REQ_START - REQ_END) * \
        #               math.exp((-1. * total_num_steps + t) / REQ_DECAY)


        # TODO, reward...
        # reward = info['reward_dist']
        epi_rewards += reward
        reward = Tensor([reward])

        # Observe new state
        if not done:
            next_state = state
        else:
            next_state = None
            # print('done!!')


        # Store the transition in memory
        # remove accumulated joint angle
        # state[0, 0] %= (2*math.pi) if state[0, 0] >= 0 else -(2*math.pi)
        # state[0, 1] %= (2*math.pi) if state[0, 1] >= 0 else -(2*math.pi)
        memory.push(current_state, action, next_state, reward)

        # Move to the next state
        current_state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            break

    total_num_steps += t
    # print('total step: ', total_num_steps, ' num_epi: ', i_episode)

    # plot
    episode_durations.append(t + 1)
    # mean10 = np.mean(episode_durations[-min(10, len(episode_durations)):])
    # print('num_epi: ', i_episode+1, '   mean10: ', mean10, '   requirement: ', requirement)
    episode_rewards.append(epi_rewards/max(t, 1))
    # episode_rewards_step += step_rewards
    # plot_reward_step()
    # episode_Err.append(np.mean(step_errs, dtype=np.float64))
    # episode_Q.append(np.mean(step_Q, dtype=np.float64))

    # plot_durations()
    # plot_reward()
    # plot_Err()
    # plot_Q()

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        # print('Target Update!!!!---------------------')
        tau = 0.05
        cp_targetParam = target_net.state_dict()
        cp_policyParam = policy_net.state_dict()

        # Soft Target Update
        for k in cp_targetParam:
            cp_targetParam[k] *= tau
            cp_policyParam[k] *= (1-tau)
            cp_targetParam[k] = cp_targetParam[k] + cp_policyParam[k]

        target_net.load_state_dict(cp_targetParam)
        # target_net.load_state_dict(policy_net.state_dict())

    # Save model at every MODEL_SAVE epi
    if i_episode % MODEL_SAVE_PERIOD == 0:
        print('model was saved..')
        save_model(i_episode)
        savePlots()

    # Log interval
    if i_episode % LOG_INTERVAL == 0:
        print('Updates {}, num steps {}, epi reward {:.2f}'.
              format(i_episode, total_num_steps, epi_rewards))



print('Complete')
input()


