import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import get_distribution


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        All classes that inheret from Policy are expected to have
        a feature exctractor for actor and critic (see examples below)
        and modules called linear_critic and dist. Where linear_critic
        takes critic features and maps them to value and dist
        represents a distribution of actions.        
        """
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        # MLP or CNN forward 호출, state는 그냥 내보냄. critic과  hidden의 마지막 output 레이어 직전의 hidden layer를 리턴
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        # pan-tilt-laser에선 action_mean, std가 각각 1x3으로 나타남
        action, action_mean, action_std = self.dist.sample(hidden_actor[-1], deterministic=deterministic)    # action을 mean과 std의 형태로 sampling해서 구함


        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor[-1], action)
        value = self.critic_linear(hidden_critic[-1])   # critic은 그냥 그대로 내보냄
        
        return value, action, action_log_probs, states, hidden_actor, hidden_critic

    def get_value(self, inputs, states, masks):        
        hidden_critic, _, states = self(inputs, states, masks)
        value = self.critic_linear(hidden_critic[-1])
        return value
    
    def evaluate_actions(self, inputs, states, masks, actions):     # input: observation, [num_step x 11]
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor[-1], actions)
        value = self.critic_linear(hidden_critic[-1])

        
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(Policy):
    def __init__(self, num_inputs, action_space, use_gru):
        super(CNNPolicy, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU()
        )
        
        if use_gru:
            self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        self.dist = get_distribution(512, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        def mult_gain(m):
            relu_gain = nn.init.calculate_gain('relu')
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.mul_(relu_gain)
    
        self.main.apply(mult_gain)

        if hasattr(self, 'gru'):
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)


        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return x, x, states


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(Policy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.hook = False
        self.action_space = action_space
        self.nNode = 128

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.nNode),
            nn.Tanh(),
            nn.Linear(self.nNode, self.nNode),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, self.nNode),
            nn.Tanh(),
            nn.Linear(self.nNode, self.nNode),
            nn.Tanh()
        )

        self.critic_linear = nn.Linear(self.nNode, 1)
        self.dist = get_distribution(self.nNode, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def set_hook_mode(self, mode):
        self.hook = mode

    def forward(self, inputs, states, masks):   # input은 observation값

        if self.hook == True:   # hook 모드가 true일 때 모든 layer에 대한 feature map을 리턴
            # 0: all
            # 1: linear(input, nNode)
            # 2: Tanh()
            # 3: linear(nNode, nNode)
            # 4: Tanh()
            a_modulelist = list(self.actor.modules())
            c_modulelist = list(self.critic.modules())

            # hidden actor feature map
            hidden_actor_list = []
            x = inputs
            for l in a_modulelist[1:]:
                x = l(x)
                hidden_actor_list.append(x)

            # hidden critic feature map
            hidden_critic_list = []
            x = inputs
            for l in c_modulelist[1:]:
                x = l(x)
                hidden_critic_list.append(x)
        else:   # hook모드가 false이면 단일 원소로 된 list를 리턴
            hidden_actor_list = [self.actor(inputs)]
            hidden_critic_list = [self.critic(inputs)]

        return hidden_critic_list, hidden_actor_list, states
