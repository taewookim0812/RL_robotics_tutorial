import numpy as np
from baselines.common.running_mean_std import RunningMeanStd
from vecEnv import VecEnv

class bVecNormalize(VecEnv):
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnv.__init__(self,
                        observation_space=venv.observation_space,
                        action_space=venv.action_space)
        print('bullet vec normalize 초기화 입니다. ')
        self.venv = venv
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)   # TODO, self.num_envs
        self.gamma = gamma
        self.epsilon = epsilon


    def step(self, action):
        return self.step_norm(action)

    def step_norm(self, action):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(action)     # 각 robot에서 정의된 step()이 호출됨
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos


    def _obfilt(self, obs):
        if self.ob_rms:
            # TODO, ret_rms가 정의되어 있지 않으면 enjoy모드로 간주하여 update안함
            self.ob_rms.update(obs) if self.ret_rms else None
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs


    def reset(self):
        obs = self.venv.reset()
        return self._obfilt(obs)

    def set_target(self, target_pos):
        self.venv.set_target(target_pos)


    def get_state(self):
        return self.venv.get_state()