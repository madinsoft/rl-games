import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding
from math import log2
from numpy.random import choice


class Base2048Env(gym.Env):

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    actions = [LEFT, UP, RIGHT, DOWN]

    ACTION_STRING = {
        LEFT: 'left',
        UP: 'up',
        RIGHT: 'right',
        DOWN: 'down',
    }

    cache = {}

    def __init__(self, width=4, height=4, cache=True):
        self.width = width
        self.height = height

        # Internal Variables
        self.board = None
        self.is_cache = cache

    def step(self, action: int):
        # """Rotate board aligned with left action"""
        # print('---------------- step')
        # print(self.ACTION_STRING[action])
        # self.render()
        # res = self.explore(action, self.board)
        # print(self.ACTION_STRING[action])
        # self.render()
        # print('================')
        # return res
        rotated_obs = np.rot90(self.board, k=action)
        reward, updated_obs, valid = self._slide_left_and_merge(rotated_obs)
        self.board = np.rot90(updated_obs, k=4 - action)

        # Place one random tile on empty location
        if valid:
            self._place_random_tiles(self.board, count=1)

        done = self.is_done()
        return self.state, reward, done, valid

    def explore(self, action: int, board):

        if self.is_cache:
            if not hasattr(board, 'flat'):
                state_in = board
                board = self.state2board(board)
            else:
                state_in = self.board2state(board)
            state_in.append(action)
            state_action = tuple(state_in)
            try:
                if len(Base2048Env.cache) > 100000:
                    Base2048Env.cache = {}
                return Base2048Env.cache[state_action]
            except KeyError:
                rotated_obs = np.rot90(board, k=action)
                reward, updated_obs, valid = self._slide_left_and_merge(rotated_obs)
                board = np.rot90(updated_obs, k=4 - action)

                if valid:
                    self._place_random_tiles(board, count=1)
                done = self.is_done(board)
                state = self.board2state(board)
                Base2048Env.cache[state_action] = state, reward, done, valid
                return Base2048Env.cache[state_action]
        else:
            if not hasattr(board, 'flat'):
                board = self.state2board(board)
            rotated_obs = np.rot90(board, k=action)
            reward, updated_obs, valid = self._slide_left_and_merge(rotated_obs)
            board = np.rot90(updated_obs, k=4 - action)

            if valid:
                self._place_random_tiles(board, count=1)
            # else:
            #     print('not valid', action)
            done = self.is_done(board)
            state = self.board2state(board)

            print('explore', state, reward, done, valid)
            return state, reward, done, valid

    def is_done(self, board=None):
        if board is None:
            copy_board = self.board.copy()
        else:
            if not hasattr(board, 'flat'):
                copy_board = self.state2board(board)
            else:
                copy_board = board.copy()

        if not copy_board.all():
            return False

        for action in [0, 1, 2, 3]:
            rotated_obs = np.rot90(copy_board, k=action)
            _, updated_obs, _ = self._slide_left_and_merge(rotated_obs)
            if not updated_obs.all():
                return False

        return True

    def reset(self):
        """Place 2 tiles on empty board."""

        self.board = np.zeros((self.width, self.height), dtype=np.int64)
        self._place_random_tiles(self.board, count=2)
        # return self.board
        return self.state

    def render(self, board=None):
        if board is None:
            board = self.board
        print('---------')
        for row in board.tolist():
            print(' \t'.join(map(str, row)))

    def is_action_legal(self, action, board=None):
        if board is None:
            board = self.board
        else:
            if not hasattr(board, 'flat'):
                board = self.state2board(board)
        _, _, _, valid = self.explore(action, board)
        return valid

    def sample(self):
        try:
            return choice(self.legal_actions)
        except ValueError:
            raise
            return 0

    def diff(self, board):
        state = self.board2state(board)
        h = abs(state[0] - state[1]) + abs(state[2] - state[3])
        v = abs(state[0] - state[2]) + abs(state[1] - state[3])
        return h + v

    def close(self, state):
        if self.width == 2 and self.height == 2:
            h = (state[0] == state[1]) + (state[2] == state[3])
            v = (state[0] == state[2]) + (state[1] == state[3])
            return h + v
        if self.width == 3 and self.height == 3:
            h1 = (state[0] == state[1]) + (state[1] == state[2])
            h2 = (state[3] == state[4]) + (state[4] == state[5])
            h3 = (state[6] == state[7]) + (state[7] == state[8])
            v1 = (state[0] == state[3]) + (state[3] == state[6])
            v2 = (state[1] == state[4]) + (state[4] == state[7])
            v3 = (state[2] == state[5]) + (state[5] == state[8])
            return h1 + h2 + h3 + v1 + v2 + v3
        return 0

    @property
    def objective_reached(self):
        if self.width == 2 and self.height == 2:
            objective = 16
        elif self.width == 3 and self.height == 3:
            objective = 128
        else:
            objective = 2048
        return self.board.max() >= objective

    @property
    def nb_free_cases(self):
        return np.count_nonzero(self.board == 0)

    @property
    def legal_actions(self):
        actions = []
        for action in self.actions:
            if self.is_action_legal(action, self.board):
                actions.append(action)
        return actions

    @property
    def state(self):
        return self.board2state(self.board)

    @state.setter
    def state(self, state):
        self.board = self.state2board(state)

    def board2state(self, board):
        return [0 if value == 0 else int(log2(value)) for value in board.flat]
        # return tuple(0 if value == 0 else int(log2(value)) for value in board.flat)

    def state2board(self, state):
        board = [0 if value == 0 else 2**value for value in state]
        return np.array(board).reshape((self.width, self.height))

    # ________________________________________________________________ private
    def _sample_tiles(self, count=1):
        """Sample tile 2 or 4."""

        # choices = [2, 4]
        choices = [2, 2]
        probs = [0.9, 0.1]

        tiles = self.np_random.choice(choices,
                                      size=count,
                                      p=probs)
        return tiles.tolist()

    def _sample_tile_locations(self, board, count=1):
        """Sample grid locations with no tile."""

        zero_locs = np.argwhere(board == 0)
        zero_indices = self.np_random.choice(len(zero_locs), size=count)

        zero_pos = zero_locs[zero_indices]
        # zero_pos = list(zip(*zero_pos))
        zero_pos = tuple(zip(*zero_pos))
        return zero_pos

    def _place_random_tiles(self, board, count=1):
        if not board.all():
            tiles = self._sample_tiles(count)
            tile_locs = self._sample_tile_locations(board, count)
            board[tile_locs] = tiles

    def _slide_left_and_merge(self, board):
        """Slide tiles on a grid to the left and merge."""

        result = []

        score = 0
        board_before = board
        for row in board:
            row = np.extract(row > 0, row)
            score_, result_row = self._try_merge(row)
            score += score_
            row = np.pad(np.array(result_row), (0, self.width - len(result_row)),
                         'constant', constant_values=(0,))
            result.append(row)
        new_board = np.array(result, dtype=np.int64)
        return score, new_board, not np.all(new_board == board_before)

    @staticmethod
    def _try_merge(row):
        score = 0
        result_row = []

        i = 1
        while i < len(row):
            if row[i] == row[i - 1]:
                score += row[i] + row[i - 1]
                result_row.append(row[i] + row[i - 1])
                i += 2
            else:
                result_row.append(row[i - 1])
                i += 1

        if i == len(row):
            result_row.append(row[i - 1])

        return score, result_row
