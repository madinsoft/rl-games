from abc import ABC, abstractmethod
from random import random


# ________________________________________________________________
class AbstractPolicy(ABC):

    def __init__(self, env, greedy=0):
        self.name = self.__class__.__name__.lower()
        self.greedy = greedy
        self.env = env

    @abstractmethod
    def _action(self, state):
        """ action to do from a specific policy """
        pass

    def action(self, state):
        if self.greedy == 0 or random() >= self.greedy:
            return self._action(state)
        else:
            return self.env.sample(state)

    def __call__(self, state):
        return self.action(state)

    def learn(self, actions, states, rewards):
        """ action to do from a specific policy """
        pass

