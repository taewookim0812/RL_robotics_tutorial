import torch.nn as nn
import torch
import torch.nn.functional as F

# CartPole DQN model
class cartpole_DQN(nn.Module):
    def __init__(self):
        super(cartpole_DQN, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.head  = nn.Linear(24, 2)

        # weight initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)



# Reacher DQN model
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



class reacher_DQN(nn.Module):
    def __init__(self):
        super(reacher_DQN, self).__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head  = nn.Linear(128, len(action_list))

        # weight initialization
        torch.nn.init.uniform_(self.fc1.weight, -0.003, 0.003)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.uniform_(self.fc2.weight, -0.003, 0.003)
        self.fc2.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)
