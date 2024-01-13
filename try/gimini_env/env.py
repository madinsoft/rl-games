import gym
from random import randint
from gym import spaces
from numpy import array as A

class GiminiEnv(gym.Env):
    def __init__(self):
        super(GiminiEnv, self).__init__()
        # self.state = [0, 0, 0, 0]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([5, 5, 5, 5])

    def step(self, action):
        min_val = min(self.state)
        if self.state[action] == min_val:
            self.state[action] += 1
        else:
            self.state[action] -= 2

        reward = 0
        terminated = False
        truncated = False
        if self.state[action] <= 0:
            terminated = True
            reward = -1
            truncated = True
        elif (self.state == [4, 4, 4, 4]).all():
            terminated = True
            reward = 1

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.state = A([randint(1, 4) for i in range(4)])
        # self.state = A([1,1,1,1])
        return self.state, options

    def render(self):
        print(f"Current state: {self.state}")
