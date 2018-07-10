import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer


class A2C_ACKTR(object):
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        # size 조정
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # 환경으로부터 실제 얻은 reward들에서 critic으로 평가한 가치를 빼서 advantage계산
        # [:-1](가장 최근 reward)은 맨 마지막 원소는 제외, 각 step별로 각각 advantage를 계산
        # 결국 num_step 각각에 대한 advantage를 계산하고, 제곱의 평균을 내어서 loss로 사용한다.
        advantages = rollouts.returns[:-1] - values     # values는 critic으로 현재 상태에 대한 가치를 평가한 것
        value_loss = advantages.pow(2).mean()           # value_loss는 critic 업데이트를 위한 loss계산


        # actor 업데이트를 위한 loss. optimization이 기본적으로 gradient decent이므로 -loss값을 계산한다.
        # delta * -log(pi(a|s))
        action_loss = -(advantages.detach() * action_log_probs).mean()


        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -  # value_loss_coef = 0.5
         dist_entropy * self.entropy_coef).backward()   # entropy_coef = 0.01

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
