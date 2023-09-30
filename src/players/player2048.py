"""
{0: 'left', 1: 'up', 2: 'right', 3: 'down'}
gym_2048.Base2048Env.ACTION_STRING[action]
(5, 8, 6.51, 0.6999285677838846)
(4, 8, 6.33, 0.8491760712596654)
/home/patrick/.virtualenvs/keras_2048/bin/python
(5, 9, 7.14, 0.8002499609497024)
(7, 9, 7.805, 0.7382242206809527)
(7, 9, 7.5, 0.8660254037844386)
(5, 9, 7.364, 0.7870857640689481)

PrudentPlayer2048
"""
# import gym_2048
import gym
from abc import ABC, abstractmethod
from numpy import array as A
import matplotlib.pyplot as plt
from math import log2
from pandas import DataFrame
import numpy as np

import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gy2048


# ________________________________________________________________
class State2048(ABC):

    env = gym.make('2048-v0', width=2, height=2)

    def __init__(self, board):
        if not hasattr(board, 'flat'):
            self.board = State2048.env.state2board(board)
        else:
            self.board = board

    def explore(self, action):
        return State2048.env.explore(action, self.board)

    def is_action_legal(self, action):
        return State2048.env.is_action_legal(action, self.board)

    @property
    def legal_actions(self):
        return State2048.env.legal_actions

    @property
    def max_value(self):
        return self.board.max()

    @property
    def log_value(self):
        return int(log2(self.max_value))

    @property
    def done(self):
        return State2048.env.is_done(self.board)

    @property
    def state(self):
        return State2048.env.board2state(self.board)


# ________________________________________________________________
class Player2048(ABC):

    def __init__(self):
        self._env = gym.make('2048-v0')
        self.reset()

    @abstractmethod
    def action(self):
        """ action to do """
        pass

    def reset(self):
        self._env.reset()
        self._actions = []
        self._rewards = []
        self._states = []
        self._top_value = None
        self.done = False

    def run(self):
        self.reset()
        done = False
        while not done:
            action = self.action()
            done = self.step(action)
            print(action, done)

    def step(self, action):
        next_state, reward, done, info = self._env.step(action)
        if info.get('valid', True):
            self._states.append(next_state)
            self._rewards.append(reward)
            self._actions.append(action)
            self._top_value = next_state.max()
            self.done = done
        return done

    def is_action_legal(self, action):
        _, _, _, info = self._env.explore(action)
        return info.get('valid', True)

    @property
    def legal_actions(self):
        actions = []
        for action in self._env.ACTION_STRING:
            if self.is_action_legal(action):
                actions.append(action)
        return actions

    @property
    def game_result(self):
        if not self._done:
            return None
        return self.top

    @property
    def run_length(self):
        return len(self._states)

    @property
    def top_value(self):
        return self._top_value

    @property
    def top(self):
        return int(log2(self._top_value))

    @property
    def state(self):
        return self._states[-1]


# ________________________________________________________________
class RandomPlayer2048(Player2048):

    def __init__(self):
        super().__init__()

    def action(self):
        return self._env.action_space.sample()

# ________________________________________________________________
class SystematicPlayer2048(Player2048):

    def __init__(self):
        super().__init__()

    def action(self):
        for action in self._env.ACTION_STRING:
            next_state, reward, done, info = self._env.explore(action)
            if info.get('valid', True):
                return action


# ________________________________________________________________
class PrudentPlayer2048(Player2048):

    def __init__(self):
        super().__init__()

    def action(self):
        env = self._env
        res = {'score': [], 'nb0': [], 'action': []}
        for action in env.ACTION_STRING:
            next_state, reward, done, info = env.explore(action)
            if info.get('valid', True):
                res['score'].append(next_state.max())
                res['nb0'].append((next_state == 0).sum())
                res['action'].append(action)
        if len(res['score']) == 0:
            print('no solution, should be done')
            env.render()
            return 0
        # df = DataFrame(res).sort_values(['score', 'nb0'], ascending=False)
        df = DataFrame(res).sort_values(['nb0', 'score'], ascending=False)
        action = df['action'].iloc[0]
        return action


# ________________________________________________________________
class GradientPlayer2048(Player2048):

    def __init__(self):
        super().__init__()

    def action(self):
        env = self._env
        scores = []
        for action in env.ACTION_STRING:
            next_state, reward, done, info = env.explore(action)
            if info.get('valid', True):
                df = DataFrame(next_state)
                df2 = df.applymap(lambda x: 0 if x == 0 else np.log2(x))
                diff1 = (df2.diff()**2).sum().sum()
                diff2 = (df2.T.diff().T**2).sum().sum()
                diff = diff1 + diff2
                scores.append((diff, action))
        scores.sort()
        return scores[0][1]


# ________________________________________________________________
class Player2048Evaluator:

    def __init__(self, Player2048, pool=None):
        self._run_lengths = None
        self._top_values = None
        self._Player2048 = Player2048
        self._pool = pool
        self._meas = Player2048.__class__.__name__

    def evaluate(self, nb_batch, batch_size):
        # run_lengths = []
        top_values = []
        for epoch in range(nb_batch):
            if self._pool:
                res = self._pool.map(self.run, range(batch_size))
            else:
                res = [self.run(i) for i in range(batch_size)]
            top_values += res

        self._top_values = A(top_values)

    def run(self, i):
        self._Player2048.run()
        print(i, self._Player2048.run_length, 2**self._Player2048.top)
        return self._Player2048.top

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.hist(self._run_lengths)
        ax = fig.add_subplot(1, 2, 2)
        ax.hist(self._top_values, 11)
        plt.show()

    def perf_top(self):
        return self._top_values.min(), self._top_values.max(), self._top_values.mean(), self._top_values.std()


# =============================================================================
if __name__ == '__main__':
    # nb_proc = multiprocessing.cpu_count() - 1
    # pool = multiprocessing.Pool(nb_proc)

    player = SystematicPlayer2048()
    bill = Player2048Evaluator(player)
    bill.evaluate(2, 2)
    print(bill.perf_top())

    # action = player.action()
    # print(action)
    # player._env.render()
    # player.step(action)
    # player._env.render()

    # # player = Prudentplayer()
    # # player = Systematicplayer()
    # # player = Gradientplayer()
    # # player.run()
    # # print(len(player._states), player._top_value)
    # bill = playerEvaluator(player)
    # bill.evaluate(2, 2)
    # print(bill.perf_top())
    # # for i in range(4):
    # #     bill.evaluate(1000)
    # #     print(bill.perf_top())

