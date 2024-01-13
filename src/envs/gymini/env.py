import gym
import gym.spaces as spaces
from random import randint
from random import random
from numpy.random import choice
from gym.utils import seeding


class BaseMiniEnv(gym.Env):

    goal = [4, 4, 4, 4]
    actions = [0, 1, 2, 3]

    def __init__(self):
        self.observation_space = spaces.Discrete(256)
        self.action_space = spaces.Discrete(4)
        self.state = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action(self, action: int):
        return self.explore(action, self.state)

    def explore(self, action: int, state=None):
        if state is None:
            state = self.state
        state = list(state)
        try:
            done = self.is_done(state)
            if done:
                if state == BaseMiniEnv.goal:
                    reward = 10
                else:
                    reward = -10
                return state.copy(), reward, done, {'valid': False}

            try:
                value = self.state[action]
            except IndexError as e:
                print(self.state, action, e)
                pass
            min_value = min(self.state)
            if value == min_value:
                reward = 1
            else:
                reward = -1 if random() > .2 else -2
            state[action] += reward

            done = self.is_done(state)
            if done:
                if state == BaseMiniEnv.goal:
                    reward = 10
                else:
                    reward = - 10

            return state.copy(), reward / 10, done, {'valid': True}
        except KeyError:
            return state.copy(), 0, done, {'valid': False}

    def is_done(self, state=None):
        state = state or self.state
        if state == BaseMiniEnv.goal:
            return True
        for elem in state:
            if elem == 0:
                return True
        return False

    def init(self):
        self.state = [randint(1, 4) for i in range(4)]
        return self.state

    def render(self):
        print(self.state)

    def is_action_legal(self, action):
        _, _, _, info = self.explore(action)
        return info.get('valid', True)

    def sample(self):
        try:
            return choice(self.legal_actions)
        except ValueError:
            return 0

    @property
    def legal_actions(self):
        actions = []
        for action in self.actions:
            if self.is_action_legal(action):
                actions.append(action)
        return actions

    @property
    def objective_reached(self):
        return self.state == BaseMiniEnv.goal
