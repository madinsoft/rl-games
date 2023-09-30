import json
from time import perf_counter
from numpy import array as A
from random import randint


root_path = '/home/patrick/projects/IA/my-2048'


def get_best_2x2_policy():
    with open(f'{root_path}/data/policy_states_2x2.json') as data_file:
        pol = json.load(data_file)

    inputs = []
    targets = []
    for inupt_str, target in pol.items():
        inp = [int(i) for i in inupt_str]
        targ = int(target)
        inputs.append(inp)
        targets.append(targ)

    inputs = A(inputs)
    targets = A(targets)
    return inputs, targets


def get_mini_policy():
    inputs = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    target = int(A(state).argmin())
                    inputs.append(state)
                    targets.append(target)

    inputs = A(inputs)
    targets = A(targets)
    return inputs, targets


def rollout(env, model, nb, limit=None, verbose=0):
    if verbose > 0:
        start = perf_counter()
    win = 0
    for i in range(nb):
        env.reset()
        state = env.state
        done = False
        length = 0
        while not done:
            length += 1
            action = model(state)
            state, reward, done, _ = env.step(action)
            if verbose > 0:
                print('rollout', i, length, state, action, reward)
            if limit and length > limit:
                done = True
                break
        if env.objective_reached:
            win += 1
    if verbose > 0:
        elapsed = perf_counter() - start
        print(f'{win}, {nb} elapsed {elapsed:.2f}s')
    return win / nb


# ____________________________________________________________ Mocks
def get_random_state():
    return [randint(0, 4) for i in range(4)]


class ModelMock:
    def predict(self, inputs):
        return get_random_state()

    def __call__(self, input):
        return randint(0, 3)


class EnvMock:
    def __init__(self, length=10):
        self.length = length
        self.count = 0

    def reset(self):
        self.count = 0

    def is_action_legal(self, action):
        return True

    def sample(self):
        return randint(0, 3)

    def step(self, action):
        self.count += 1
        done = self.count > self.length
        return get_random_state(), randint(0, 3), done, {'valid': True}

    @property
    def objective_reached(self):
        return randint(0, 4) == 2


