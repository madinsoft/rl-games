import gym
from random import randint
from gym import spaces
from numpy import array as A
from utils.alea import Sampler


sampler = Sampler([1,2], [2,1])


class GiminiEnv(gym.Env):
    def __init__(self):
        super(GiminiEnv, self).__init__()
        # self.state = [0, 0, 0, 0]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([5, 5, 5, 5])

    def step(self, action):
        self.state, reward, terminated, truncated, options = GiminiEnv.explore(self.state, action)
        return self.state, reward, terminated, truncated, options
    
    def explore(self, action, state=None):
        state = state if state is not None else self.state
        state = state.copy()
        min_val = min(state)
        if state[action] == min_val:
            state[action] += 1
        else:
            state[action] -= sampler()

        reward = 0
        terminated = False
        truncated = False
        if state[action] <= 0:
            terminated = True
            reward = -1
            truncated = True
        elif (state == [4, 4, 4, 4]).all():
            terminated = True
            reward = 1

        return state, reward, terminated, truncated, {}

    def is_action_legal(self, action, state=None):
        _, _, _, truncated, _ = self.explore(action, state)
        return not truncated

    def sample_legal_action(self, state=None):
        while True:
            action = self.action_space.sample()
            if self.is_action_legal(action, state):
                return action

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.state = A([randint(1, 4) for i in range(4)])
        return self.state, options

    def render(self):
        print(f"Current state: {self.state}")
