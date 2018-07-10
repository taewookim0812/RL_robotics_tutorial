import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AddBias


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x, dim=1)
        if deterministic is False:
            action = probs.multinomial(num_samples=1)
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


# distribution이 각 parameter별로 독립적이라고 가정할 때 Diagonal Gaussian을 사용
# 반대로 parameter들이 서로 종속적이면 일반적인 Multivariate Gaussian Distribution을 사용.
# 이때 평균은 같으나, 분산은 corvariance matrix사용
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):        # input: 64, out: 2
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)   # actor의 마지막 layer가 여기 있음
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)   # hidden_actor를 받아서 action을 리턴 (j1 mean, j2 mean)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)  # bias를 더함. (KFAC 알고리즘을 위한 건데.. 잘 모르겠다 )
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)    # x는 hidden_actor(actor 마지막 전 layer), 바로 위의 forward호출

        action_std = action_logstd.exp()

        if deterministic is False:  # Continuous
            noise = torch.randn(action_std.size())
            if action_std.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise   # 일단 나온 action에 standard dev와 noise를 더해서 리턴(continuous하니까 이렇게 하는 듯?)
        else:
            action = action_mean
        return action, action_mean, action_std  # TODO, action_mean과 action_std 추가 했음

    # normal distribution이라 가정하고, 실제로 한 action들에 대해서 log probability를 구하는 것.
    def logprobs_and_entropy(self, x, actions):
        action_mean, action_logstd = self(x)        # 역시 actor의 마지막 layer를 통해 action_mean, logstd를 구한다.

        action_std = action_logstd.exp()            # 구한 log단위의 std를 exp를 통해 일반적인 std로 만들고

        # 아래 식은 normal distribution의 log버전. normal dist. 식에 log를 취해서 정리하면 아래와 같이 나온다.
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(-1, keepdim=True)   # 원래 곱하기인데 log니까 sum해서 더해준 듯

        # normal distribution에 대한 entropy
        # action에 대한 std가 entropy(불확실성)에 영향을 미침
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        dist_entropy = dist_entropy.sum(-1).mean()  # mean은 당장은 의미가 없을 것 같은데.. scalar에 mean하니까..
        return action_log_probs, dist_entropy


def get_distribution(num_inputs, action_space):
    if action_space.__class__.__name__ == "Discrete":
        num_outputs = action_space.n
        dist = Categorical(num_inputs, num_outputs)
    elif action_space.__class__.__name__ == "Box":
        num_outputs = action_space.shape[0]
        dist = DiagGaussian(num_inputs, num_outputs)    # input=64
    else:
        raise NotImplementedError
    return dist
