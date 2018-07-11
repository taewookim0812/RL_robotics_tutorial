import argparse
import os
import time

import numpy as np
import torch

import bullet_env as blt
import matplotlib.pyplot as plt
plt.switch_backend('agg')


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
args = parser.parse_args()

# TODO --------------------------------
# User defined parameter
args.num_stack = 1
args.env_name = 'pybullet_reacher'
args.algo = 'ppo'
args.vis = True
args.port = 8097
# args.add_timestep = True
# TODO --------------------------------

nWin = 4
if args.vis:
    from visdom import Visdom

    viz = Visdom(port=args.port)
    win = [None for i in range(nWin)]


env = blt.bullet_env('GUI', args.env_name, 100, enjoy_mode=True)

actor_critic, ob_rms = \
    torch.load(os.path.join(args.load_dir + args.algo + '/', args.env_name + ".pt"))


net_param = actor_critic.state_dict()


print("model: ", actor_critic)
print('ob_rms: ', ob_rms)
print('net_param : ', net_param)
print('net keys: ', net_param.keys())

if len(env.robot_dict[args.env_name].observation_space.shape) == 1:
    # env = VecNormalize(env, ret=False)
    env.robot_dict[args.env_name].ob_rms = ob_rms




obs_shape = (env.robot_dict[args.env_name].observation_space.shape[0], )   # tuple
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)


def update_current_obs(obs):
    shape_dim0 = env.robot_dict[args.env_name].observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


obs = env.reset()
update_current_obs(obs)
# actor_critic.set_hook_mode(True)
t = 0

prev_q1 = 0
prev_q2 = 0

current_q1 = 0
current_q2 = 0
alpha = 0.8
while True:
    with torch.no_grad():
        value, action, _, states, hidden_actor, hidden_critic = actor_critic.act(current_obs,
                                                    states,
                                                    masks, deterministic=True)
    cpu_actions = action.squeeze(1).cpu().numpy()

    current_q1 = cpu_actions[0][0]
    current_q2 = cpu_actions[0][1]

    cpu_actions[0][0] = current_q1 * alpha + prev_q1 * (1 - alpha)
    cpu_actions[0][1] = current_q2 * alpha + prev_q2 * (1 - alpha)

    # cpu_actions = [[0, 0, 0.0001]]

    # Obser reward and next obs
    rst = env.step(cpu_actions[0])
    obs, reward, done, info = rst[args.env_name]


    prev_q1 = current_q1
    prev_q2 = current_q2


    dist_err = info[0]['reward_dist']
    ctrl_err = info[0]['reward_ctrl']


    if abs(dist_err) < 0.05 and abs(ctrl_err) < 0.0001:
        print('convergence!!')

    masks.fill_(0.0 if done else 1.0)

    if current_obs.dim() == 4:
        current_obs *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_obs *= masks
    update_current_obs(obs)

    time.sleep(0.02)
    t += 1



