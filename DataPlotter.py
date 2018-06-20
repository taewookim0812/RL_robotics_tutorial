import matplotlib.pyplot as plt
import torch


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
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


episode_rewards = []
def plot_reward():
    plt.figure(3)
    plt.clf()
    reward_t = torch.FloatTensor(episode_rewards)
    plt.title('Episode Mean Reward Plot')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


episode_rewards_step = []
def plot_reward_step():
    plt.figure(3)
    plt.clf()
    reward_t = torch.FloatTensor(episode_rewards_step)
    plt.title('Reward Plot at every step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


step_Q = []
episode_Q = []
def plot_Q():
    plt.figure(4)
    plt.clf()
    Q_t = torch.FloatTensor(episode_Q)
    plt.title('Episode Mean Q Plot')
    plt.xlabel('Episode')
    plt.ylabel('Q')
    plt.plot(Q_t.numpy())
    if len(Q_t) >= 100:
        means = Q_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)


episode_Err = []
def plot_Err():
    plt.figure(5)
    plt.clf()
    Err_t = torch.FloatTensor(episode_Err)
    plt.title('Episode Mean Error')
    plt.xlabel('Episode')
    plt.ylabel('Pos Err')
    plt.plot(Err_t.numpy())
    if len(Err_t) >= 100:
        means = Err_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)