from policies.abstract_policy import AbstractPolicy
import numpy as np
from random import random


# ________________________________________________________________
class PolicyRandom:

    def __init__(self, env):
        self.name = self.__class__.__name__.lower()
        self.env = env

    def action(self, state):
        """ action to do from a specific policy """
        return self.env.sample(state)

    def __call__(self, state):
        return self.action(state)


# ________________________________________________________________
class PolicyMinimizeNbTile(AbstractPolicy):

    def _action(self, state):
        """ action to do from a specific policy """
        min_nb_tiles = self.env.length + 1
        min_action = None
        for action in self.env.legal_actions(state):
            new_state, _, _, _ = self.env.explore(action, state)
            nb_tile = np.count_nonzero(new_state)
            if nb_tile < min_nb_tiles:
                min_nb_tiles = nb_tile
                min_action = action
        if min_action is None:
            print('state', state)
            print('actions', self.env.legal_actions(state))
            print('min_nb_tiles', min_nb_tiles)
            print('nb_tile', nb_tile)
            print('min_action', min_action)

        return min_action


# ________________________________________________________________
class PolicyMaximizeReward(AbstractPolicy):

    def _action(self, state):
        """ action to do from a specific policy """
        max_reward = -1
        max_action = None
        for action in self.env.legal_actions(state):
            _, reward, _, _ = self.env.explore(action, state)
            if reward > max_reward:
                max_reward = reward
                max_action = action
        return max_action


# ________________________________________________________________
class PolicyMaximizeRewardOnCorner(AbstractPolicy):

    def _action(self, state):
        """ action to do from a specific policy """
        max_reward = -1
        max_action = None
        actions = self.env.legal_actions(state)
        for action in actions:
            new_state, reward, _, _ = self.env.explore(action, state)
            if new_state.max() == new_state[0]:
                if reward > max_reward:
                    max_reward = reward
                    max_action = action
        if max_action:
            return max_action
        return actions[0]

# ________________________________________________________________
class PolicyMaximizeSameNeighbors(AbstractPolicy):

    def _action(self, state):
        """ action to do from a specific policy """
        max_nb = -1
        max_action = None
        for action in self.env.legal_actions(state):
            new_state, _, _, _ = self.env.explore(action, state)
            nb_same_neighbors = self.env.nb_identical_neighbors(new_state)
            if nb_same_neighbors > max_nb:
                max_nb = nb_same_neighbors
                max_action = action
        return max_action


# ________________________________________________________________
class PolicyMixte(AbstractPolicy):

    def __init__(self, env, w_rand=1, w_free=1, w_reward=1, w_neighbors=1, greedy=0):
        super().__init__(env, greedy)
        self.w_rand = w_rand
        self.w_free = w_free
        self.w_reward = w_reward
        self.w_neighbors = w_neighbors
        self.pol_rand = PolicyRandom(env)
        self.pol_free = PolicyMinimizeNbTile(env)
        self.pol_reward = PolicyMaximizeReward(env)
        self.pol_neighbors = PolicyMaximizeSameNeighbors(env)

    def _action(self, state):
        """ action to do from a specific policy """
        actions = np.zeros(4)
        actions[self.pol_rand(state)] += self.w_rand + random() / 10
        actions[self.pol_free(state)] += self.w_free + random() / 10
        actions[self.pol_reward(state)] += self.w_reward + random() / 10
        actions[self.pol_neighbors(state)] += self.w_neighbors + random() / 10
        # print(actions, actions.argmax())
        return actions.argmax()


# ===============================
if __name__ == '__main__':
    from envs.game_2048 import Game2048
    from models.cbs import Roller
    from time import perf_counter
    from numpy import array as A

    env = Game2048(size=3, cache=True)
    # pol = PolicyRandom(env)
    # pol = PolicyMinimizeNbTile(env)
    # pol = PolicyMaximizeReward(env)
    pol = PolicyMaximizeSameNeighbors(env)
    # for policy in [PolicyRandom, PolicyMinimizeNbTile, PolicyMaximizeReward, PolicyMaximizeSameNeighbors]:
    # for weights in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
    # for weights in [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (.5, .5, .5, .9)]:
    # for weights in [(.5, .5, .5, 1.5)]:
    # for i in range(10):
    #     pol = PolicyMixte(env, *weights)
    #     print(weights)

    for i in range(10):
        roller = Roller(env, pol, nb=100, verbose=False)
        cronos = perf_counter()
        wins = roller()
        elapsed = perf_counter() - cronos
        max_states = A(roller.max_states)
        print('   ', wins, roller.reward, max_states.max(), max_states.mean(), f'elapsed {elapsed:.2f} seconds')
