
class VecEnv:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        pass

    def step_norm(self, actions):
        pass

    def step(self, actions):
        self.step_norm(actions)
        pass

    def reset(self):
        pass