from time import perf_counter
from numpy import array as A


# ________________________________________________________________
class Runner:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def __call__(self, model, env):
        done = False
        state = env.reset()
        self.states = [state]
        self.actions = []
        self.rewards = []
        while not done:
            action = model(state)
            self.actions.append(action)
            state, reward, done, infos = env.step(action)
            self.states.append(state)
            self.rewards.append(reward)
        return max(state)

    @property
    def reward(self):
        return sum(self.rewards)


# ____________________________________________________________ Roller
class Roller:

    def __init__(self, env, model, percent=100., nb=20, limit=None, verbose=0):
        self.env = env
        self.model = model
        self.percent = percent
        self.nb = nb
        self.limit = limit
        self.verbose = verbose
        self.reward = 0
        self.max_states = []
        self.wins = 0

    def __call__(self, model=None):
        model = self.model
        nb = self.nb
        if self.verbose > 0:
            start = perf_counter()
        win = 0
        mean_reward = 0
        self.max_states = []
        for i in range(nb):
            state = self.env.reset()
            done = False
            length = 0
            # print(f'Roller {i} {length} {self.limit}')
            sum_reward = 0
            while not done:
                length += 1
                action = model(state)
                # if not self.env.is_action_legal(action, self.env.board):
                if not self.env.is_action_legal(action):
                    action = self.env.sample()
                state, reward, done, _ = self.env.step(action)
                if self.verbose >= 3:
                    print(max(state), 'cbs rollout', i, length, state, action, reward)
                if self.limit and length > self.limit:
                    done = True
                    break
                sum_reward += reward
            mean_reward += sum_reward
            max_state = max(self.env.state)
            self.max_states.append(max_state)
            if self.env.objective_reached:
                win += 1
            if self.verbose >= 2:
                print(f'{i} roller length: {length} max_state: {max_state} sum_reward:{sum_reward}')
        if self.verbose >= 1:
            elapsed = perf_counter() - start
            print(f'{win}, {nb} elapsed {elapsed:.2f}s')
        # return (win / nb) * self.percent, mean_reward / nb
        self.reward = mean_reward / nb
        # self.max_states = max_state / nb
        self.wins = win / nb
        # li = len(self.env.cache)
        # print(f'Roller env cache {li}')
        return self.wins * self.percent


# ____________________________________________________________ Evaluator
class Evaluator:

    def __init__(self, states, targets, model=None, percent=100.):
        self.states = states
        self.targets = targets
        self.model = model
        self.percent = percent

    def __call__(self, model=None, verbose=0):
        model = self.model
        if model is None:
            return 0
        outs = self.model.predict(self.states)
        count = 0
        for target, output in zip(self.targets, outs):
            if verbose > 0:
                print('Evaluator', target, output, count)
            if not hasattr(target, '__iter__') and not hasattr(output, '__iter__'):
                target = A([target])
                output = A([output])
            elif not hasattr(target, '__iter__') and hasattr(output, '__iter__'):
                target = A([target])
            elif hasattr(target, '__iter__') and not hasattr(output, '__iter__'):
                output = A([output])
            if (output == target).all():
                count += 1
            # else:
            #     print(f'output {output} target {target}')
        return (count / len(self.states)) * self.percent


# ===============================
if __name__ == '__main__':
    from mocks import ModelMock
    from mocks import EnvMock
    from mocks import get_random_state
    from mocks import get_random_action
    # ____________________________________________________________
    nb = 10
    model = ModelMock()
    env = EnvMock()
    roller = Roller(env, model)
    inputs = [get_random_state() for j in range(nb)]
    targets = [get_random_action() for i in range(len(inputs))]
    avalor = Evaluator(inputs, targets, model)
    coverage = avalor()
    print(f'coverage {coverage}%')
    wins, reward = roller(10)
    print(f'wins {wins}%, reward {reward}')

