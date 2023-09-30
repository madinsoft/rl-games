import gym
from tqdm import tqdm
# import json
# from sklearn.metrics import accuracy_score
from tools.sdi_vizu import SdiVizu
# ___________________________________________ project libs
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gy2048
from policies.policy_ql import PolicyQL
from utils import get_best_2x2_policy
from utils import Evaluator
from utils import Roller


# ___________________________________________ class
class Best2x2Policy:

    def __init__(self):
        self.env = gym.make('2048-v0', width=2, height=2)

    def __call__(self, state):
        self.env.state = state
        best = []
        for action in self.env.legal_actions:
            state, reward, done, infos = env.explore(action)
            close = self.env.close(state)
            best.append((reward + close / 10, action))

        best.sort(key=lambda x: x[0])
        return best[-1][1]


# ============================================= main
best_model = Best2x2Policy()
# env = gym.make('2048-v0', width=2, height=2)
# roller = Roller(env, model)
# success, mean_reward = roller(10, verbose=True)
# print(success, mean_reward)

inputs, targets = get_best_2x2_policy()
viz = SdiVizu('qv', 'success', 'cov', 'mean', dt=4, measurement='deepQ', model_name='2x2', clear=True)

env = gym.make('2048-v0', width=2, height=2)
pol = PolicyQL(env, greedy=0, learning=1, actualisation=1)
roller = Roller(env, pol)
evaluator = Evaluator(inputs, targets, pol)
# for i in range(1000):
for i in tqdm(range(10000)):
    done = False
    state = env.reset()
    states = []
    actions = []
    rewards = []
    states.append(state)
    while not done:
        # if i < 500:
        #     action = env.sample()
        # else:
        #     action = pol.action(state)
        action = best_model(state)
        # action = env.sample()
        actions.append(action)
        state, reward, done, infos = env.step(action)
        states.append(state)
        rewards.append(reward)
    pol.learn(states, actions, rewards)
    if i % 100 == 0:
        cov = evaluator(pol) * 100
        success, mean_reward = roller(100)
        success *= 100
        size = len(pol.Q)
        q_values = pol.mean
        print(f'{i} size {size} {q_values:.2f} coverage: {cov:.2f}% success: {success:.2f}% mean reward {mean_reward}')
        viz(q_values, success, cov, mean_reward)

cov = evaluator(pol) * 100
success, mean_reward = roller(100)
success *= 100
size = len(pol.Q)
q_values = pol.mean
print(f'{i} size {size} {q_values:.2f} coverage: {cov:.2f}% success: {success:.2f}% mean reward {mean_reward}')
viz(q_values, success, cov, mean_reward)
