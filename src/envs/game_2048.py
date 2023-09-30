import numpy as np
from numpy import array as A
from random import randint
from numpy.random import choice


class Game2048:

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    actions_to_str = {
        LEFT: 'left',
        UP: 'up',
        RIGHT: 'right',
        DOWN: 'down',
    }

    cache = {}
    cache_limit = 100000

    def __init__(self, size=4, cache=True):
        self.size = size
        self.length = size * size
        if cache:
            self.explore = self.explore_with_cache
        else:
            self.explore = self._explore
        self.state = self.reset()

    def reset(self, cache=True):
        self.state = np.zeros(self.length, dtype=np.int64)
        self.state[randint(0, self.length - 1)] = 1
        if cache:
            Game2048.clear_cache()
        return self.state

    def step(self, action):
        self.state, reward, done, info = self.explore(action, self.state)
        return self.state, reward, done, info

    def explore(self, action, state=None):
        """
        choose explore_with_cache if cache is activated
        """
        pass

    def explore_with_cache(self, action, state=None):
        if state is None:
            state = self.state
        if len(Game2048.cache) > Game2048.cache_limit:
            Game2048.cache = {}
        state_tuple = tuple(state)
        state_action_tuple = (state_tuple, action)
        if state_action_tuple not in Game2048.cache:
            Game2048.cache[state_action_tuple] = self._explore(action, state)
        return Game2048.cache[state_action_tuple]

    def _explore(self, action, state=None):
        if state is None:
            state = self.state
        new_state = A(state)
        score = 0
        is_same_state = True
        nb_zeros_all = 0
        # ================================================================== LEFT
        if action == Game2048.LEFT:
            for line in range(self.size):
                offset = line * self.size
                line_state = A(state[offset:offset + self.size], dtype=np.int64)
                if not line_state.any():
                    continue

                row = np.zeros(self.size, dtype=np.int64)
                # ________________________________________________________________ displace
                i = 0
                j = 0
                nb_zeros_line = 0
                while i < self.size:
                    if line_state[i] == 0:
                        i += 1
                        nb_zeros_all += 1
                        nb_zeros_line += 1
                    else:
                        row[j] = line_state[i]
                        i += 1
                        j += 1
                        if nb_zeros_line > 0:
                            is_same_state = False

                # ________________________________________________________________ merge
                i = 0
                j = 0
                line_state = row
                row = np.zeros(self.size, dtype=np.int64)
                while i < self.size:
                    if i == self.size - 1:
                        row[j] = line_state[i]
                        i += 1
                    elif line_state[i] != 0 and line_state[i] == line_state[i + 1]:
                        gain = line_state[i] * 2
                        score += gain
                        row[j] = line_state[i] + 1
                        i += 2
                        is_same_state = False
                    else:
                        row[j] = line_state[i]
                        i += 1
                    j += 1

                if not is_same_state:
                    i = offset
                    for value in row:
                        new_state[i] = value
                        i += 1
        # ================================================================== RIGHT
        elif action == Game2048.RIGHT:
            for line in range(self.size):
                offset = line * self.size
                line_state = A(state[offset:offset + self.size], dtype=np.int64)
                if not line_state.any():
                    continue

                row = np.zeros(self.size, dtype=np.int64)
                # ________________________________________________________________ displace
                i = self.size - 1
                j = self.size - 1
                nb_zeros_line = 0
                while i >= 0:
                    if line_state[i] == 0:
                        i -= 1
                        nb_zeros_all += 1
                        nb_zeros_line += 1
                    else:
                        row[j] = line_state[i]
                        i -= 1
                        j -= 1
                        if nb_zeros_line > 0:
                            is_same_state = False

                # ________________________________________________________________ merge
                i = self.size - 1
                j = self.size - 1
                line_state = row
                row = np.zeros(self.size, dtype=np.int64)
                while i >= 0:
                    if i == 0:
                        row[j] = line_state[i]
                        i -= 1
                    elif line_state[i] != 0 and line_state[i] == line_state[i - 1]:
                        gain = line_state[i] * 2
                        score += gain
                        row[j] = line_state[i] + 1
                        i -= 2
                        is_same_state = False
                    else:
                        row[j] = line_state[i]
                        i -= 1
                    j -= 1

                if not is_same_state:
                    i = offset
                    for value in row:
                        new_state[i] = value
                        i += 1
        # ================================================================== UP
        elif action == Game2048.UP:
            for col in range(self.size):
                col_state = A(state[col::self.size], dtype=np.int64)
                if not col_state.any():
                    continue

                row = np.zeros(self.size, dtype=np.int64)
                # ________________________________________________________________ displace
                i = 0
                j = 0
                nb_zeros_line = 0
                while i < self.size:
                    if col_state[i] == 0:
                        i += 1
                        nb_zeros_all += 1
                        nb_zeros_line += 1
                    else:
                        row[j] = col_state[i]
                        i += 1
                        j += 1
                        if nb_zeros_line > 0:
                            is_same_state = False

                # ________________________________________________________________ merge
                i = 0
                j = 0
                col_state = row
                row = np.zeros(self.size, dtype=np.int64)
                while i < self.size:
                    if i == self.size - 1:
                        row[j] = col_state[i]
                        i += 1
                    elif col_state[i] != 0 and col_state[i] == col_state[i + 1]:
                        gain = col_state[i] * 2
                        score += gain
                        row[j] = col_state[i] + 1
                        i += 2
                        is_same_state = False
                    else:
                        row[j] = col_state[i]
                        i += 1
                    j += 1

                if not is_same_state:
                    i = col
                    for value in row:
                        new_state[i] = value
                        i += self.size
        # ================================================================== DOWN
        elif action == Game2048.DOWN:
            for col in range(self.size):
                col_state = A(state[col::self.size], dtype=np.int64)
                if not col_state.any():
                    continue

                row = np.zeros(self.size, dtype=np.int64)
                # ________________________________________________________________ displace
                i = self.size - 1
                j = self.size - 1
                nb_zeros_line = 0
                while i >= 0:
                    if col_state[i] == 0:
                        i -= 1
                        nb_zeros_all += 1
                        nb_zeros_line += 1
                    else:
                        row[j] = col_state[i]
                        i -= 1
                        j -= 1
                        if nb_zeros_line > 0:
                            is_same_state = False

                # ________________________________________________________________ merge
                i = self.size - 1
                j = self.size - 1
                col_state = row
                row = np.zeros(self.size, dtype=np.int64)
                while i >= 0:
                    if i == 0:
                        row[j] = col_state[i]
                        i -= 1
                    elif col_state[i] != 0 and col_state[i] == col_state[i - 1]:
                        gain = col_state[i] * 2
                        score += gain
                        row[j] = col_state[i] + 1
                        i -= 2
                        is_same_state = False
                    else:
                        row[j] = col_state[i]
                        i -= 1
                    j -= 1

                if not is_same_state:
                    i = col
                    for value in row:
                        new_state[i] = value
                        i += self.size

        valid = not is_same_state
        if valid:
            new_state = self._place_random_tiles(new_state)
        return new_state, score, self.is_done(new_state), {'valid': not is_same_state}

    def is_done(self, state=None):
        if state is None:
            state = self.state
        if not A(state).all():
            return False
        return self.nb_identical_neighbors(state) == 0

    def legal_actions(self, state=None):
        if state is None:
            state = self.state
        actions = []
        for action in self.actions_to_str:
            if self.is_action_legal(action, state):
                actions.append(action)
        return actions

    def is_action_legal(self, action, state=None):
        if state is None:
            state = self.state
        _, _, _, info = self.explore(action, state)
        return info['valid']

    def render(self):
        for i in range(self.size):
            offset = i * self.size
            print(' '.join(str(e) for e in self.state[offset:offset + self.size]))

    @property
    def objective_reached(self):
        if self.size == 2:
            objective = 4
        elif self.size == 3:
            objective = 7
        else:
            objective = 11
        return self.state.max() >= objective

    def nb_identical_neighbors(self, state=None):
        if state is None:
            state = self.state

        # ________________________________ count nb horizontal neighbors
        h = 0
        step = self.size
        lines = list(range(step))
        cols = lines[:-1]
        for line in lines:
            offset = line * step
            for col in cols:
                n0 = col + offset
                n1 = col + 1 + offset
                if state[n0] != 0 and state[n0] == state[n1]:
                    h += 1

        # ________________________________ count nb vertical neighbors
        v = 0
        cols, lines = lines, cols
        for col in cols:
            for line in lines:
                n0 = line * step + col
                n1 = (line + 1) * step + col
                if state[n0] != 0 and state[n0] == state[n1]:
                    v += 1
        return h + v

    def sample(self, state=None):
        if state is None:
            state = self.state
        actions = self.legal_actions(state)
        if actions:
            return choice(actions)
        return None

    def _place_random_tiles(self, state=None):
        if state is None:
            state = self.state
        locations = np.where(state == 0)[0]
        if len(locations) > 0:
            state[choice(locations)] = 1
        return state

    @classmethod
    def clear_cache(cls):
        cls.cache = {}


# =================================================================
if __name__ == '__main__':
    env = Game2048(size=3)
    # print(env.is_done([2, 3, 2, 3, 2, 4, 1, 5, 1]))

    count = 0
    for i in range(2):
        state = env.reset()
        print('----------------- starting')
        env.render()
        done = False
        while not done:
            action = env.sample(state)
            if action is None:
                print(state.tolist())
                break
            action_str = Game2048.actions_to_str[action]
            state, reward, done, info = env.step(action)
            print('-----------------', action_str, reward)
            env.render()
        if env.objective_reached:
            count += 1
        print('state max', env.state.max())
    print(f'wins: {count}')

    # action = Game2048.DOWN
    # for i in range(env.length):
    #     state = np.zeros(env.length)
    #     state[i] = 1
    #     state[(i + env.size) % env.length] = 1
    #     state[(i + 2) % env.length] = 1
    #     stati, r, d, info = env.explore(action, state)
    #     print(state)
    #     print(stati, r, d, info)
    #     print('--------------------------------')

    # state = A([1, 0, 1, 2, 1, 2, 1, 2, 1], dtype=np.int64)
    # print(env.nb_identical_neighbors(state))
    # print(env.is_done(state))
    # print(env.explore(state, Game2048.DOWN))
    # print(env.explore(state, Game2048.LEFT))
    # print(env.explore(state, Game2048.RIGHT))
    # print(env.explore(state, Game2048.UP))

    # stati, r, d, info = env.explore(state, action)
    # stati, r, d, info = env.explore(state, action)
    # stati, r, d, info = env.explore(state, action)
    # print(state)
    # print(stati, r, d, info)


