import site
from numpy import array as A
# from time import perf_counter
import gym
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gy2048
from tools.sdi_vizu import SdiVizu
from players.player_mini import PlayerMini
from policies.policy_ql import PolicyQL
from utils import get_best_2x2_policy
from utils import Roller


# ________________________________________________________________
def eval_action(env, action):
    state, reward, done, infos = env.explore(action)
    close = env.close(state)
    return reward + close / 10


# ___________________________________________ class
class Best2x2Policy:

    def __init__(self):
        self.env = gym.make('2048-v0', width=2, height=2)

    def __call__(self, state):
        self.env.state = state
        best = []
        for action in self.env.legal_actions:
            note = eval_action(self.env, action)
            best.append((note, action))
            # state, reward, done, infos = env.explore(action)
            # close = self.env.close(state)
            # best.append((reward + close / 10, action))

        best.sort(key=lambda x: x[0])
        return best[-1][1]


class Evaluator:

    def __init__(self, states, targets):
        self.states = states
        self.targets = targets

    def round(self, value):
        distances = (self.possible_values - value)**2
        closest = distances.argmin()
        return self.possible_values[closest]

    def __call__(self, model):
        outs = model.predict(self.states)
        count = 0
        for state, target, output in zip(self.states, self.targets, outs):
            # if output != -1:
            #     print(f'evaluator state {state} target {target} output {output}')
            if target == output:
                count += 1
            # try:
            #     value = A([self.round(i) for i in output])
            # except TypeError:
            #     value = A([self.round(output)])
            # if (value == target).all():
            #     count += 1
        return count / len(self.states)


# ================================
space = 1000
roll_depth = 300
env = gym.make('2048-v0', width=2, height=2)
env_pol = gym.make('2048-v0', width=2, height=2)
polQB = PolicyQL(env=env_pol, greedy=0)
done = False
length = 0

viz = SdiVizu('coverage', 'success', 'mean_reward', dt=4, measurement='deepQ', clear=True)

best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()
evaluator = Evaluator(inputs, targets)

roller = Roller(env_pol, polQB)

for i in range(space):
    states = []
    actions = []
    rewards = []
    env.seed(i)
    state = env.reset().copy()
    states.append(state)
    done = False
    while not done:
        # action = best_model(state)
        action = env.sample()
        next_state, reward, done, info = env.step(action)
        # print(f'state {state}, action {action}, next_state {next_state}, reward {reward}')
        actions.append(action)
        rewards.append(reward)
        state = next_state
        states.append(state)
    polQB.learn(states, actions, rewards)

    cov = evaluator(polQB) * 100
    success, mean_reward = roller(100)
    success *= 100
    print(f'{i} coverage {cov:.2f}% success {success:.2f}% mean_reward {mean_reward}')
    viz(cov, success, mean_reward)
    if success > 99:
        break

polQB.save('/home/patrick/projects/IA/my-2048/data/q_2x2_3.json')
