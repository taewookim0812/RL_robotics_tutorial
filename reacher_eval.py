import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T

import importlib
# import myUtil as ut
import ModelDescriptor as md

env = gym.make('Reacher-v2').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()



# if gpu is to be used
use_cuda = torch.cuda.is_available()

if use_cuda:
    print('cuda is available')
else:
    print('cuda is NOT available')

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training(Simple)...')
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



# [q1, q2, q1_dot, q2_dot, fx, fy, fz, tx, ty, tz]
# def get_state():
#     theta = env.sim.data.qpos.flat[:2]
#     theta_dot = env.sim.data.qvel.flat[:2]
#     finger = env.get_body_com("fingertip")
#     target = env.get_body_com("target")
#     state = np.concatenate((theta, theta_dot, finger, target))
#     state = np.ascontiguousarray(state, dtype=np.float64)
#     return torch.from_numpy(state).unsqueeze(0).type(Tensor)


def get_state(state):
    state = np.ascontiguousarray(state, dtype=np.float64)
    state = torch.from_numpy(state).unsqueeze(0).type(
        torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    return state


# from action index to joint velocity
def action_interpreter(action_index):
    return np.array(md.action_list[int(action_index[0])])


# Load Reacher Trained model
nEpi = 31000
loadPath = './model/Reacher'
modelName = '/ReacherModel' + '_epi' + str(nEpi) + '.pt'



# loadPath = 'ReacherModel(Simple)_' + 'epi' + str(nEpi) + '_i10_400_300_o25_' + '.pt'
importlib.reload(md)
trainedModel = md.reacher_DQN()
print(trainedModel)
trainedModel.load_state_dict(torch.load(loadPath + modelName))
trainedModel.eval()

if use_cuda:
    trainedModel.cuda()

print('Reacher Evaluation code')

numEpisode = 1000
for i in range(numEpisode):
    state = env.reset()
    print('episode: ', i)
    for j in range(300):
        state = get_state(state)
        # action = trainedModel(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        with torch.no_grad():
            action = trainedModel(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)

        JointVel = action_interpreter(action)

        # step
        state, reward, done, _ = env.step(JointVel)


        env.render()

        err = np.linalg.norm(env.get_body_com("target") - env.get_body_com("fingertip"))
        qvel = env.sim.data.qvel.flat[:2]
        # if err < 0.04 and np.square(qvel).sum() < 0.02:
        #     done = True

        # print(err)
        if done:
            # episode_durations.append(j+1)
            # plot_durations()
            break

print(trainedModel.parameters())


