import gym
from tqdm import tqdm
# import json
from tools.sdi_vizu import SdiVizu
# ___________________________________________ project libs
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gy2048
from policies.policy_ql import PolicyQL
from utils import get_best_2x2_policy
from models.cbs import Evaluator
from models.cbs import Roller
# from policies.best_close import Best2x2Policy
from policies.policy_close_best import PolicyCloseBest


# ============================================= main
# best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()
viz = SdiVizu('qv', 'success', 'cov', dt=4, measurement='deepQ', model_name='2x2')

env = gym.make('2048-v0', width=2, height=2, cache=True)
pol = PolicyQL(env, greedy=0, learning=.9, actualisation=.9)
roller = Roller(env, pol)
evaluator = Evaluator(inputs, targets, pol)
best_model = PolicyCloseBest(env)

for i in range(10000):
    done = False
    state = env.reset()
    states1 = []
    actions = []
    rewards = []
    states1.append(state)
    while not done:
        action = best_model(state)
        actions.append(action)
        state, reward, done, infos = env.step(action)
        states1.append(state)
        rewards.append(reward)
        # print(state, action, reward)
    pol.learn(states1, actions, rewards)
    if i % 2 == 0:
        cov = evaluator()
        success = roller()
        size = len(pol.Q)
        q_values = pol.mean
        print(f'{i} size {size} {q_values:.2f} coverage: {cov:.2f}% success: {success:.2f}%')
        viz(q_values, success, cov)
        if success == 100:
            break

cov = evaluator(pol)
success = roller(100)
size = len(pol.Q)
q_values = pol.mean
print(f'{i} size {size} {q_values:.2f} coverage: {cov:.2f}% success: {success:.2f}% ')
viz(q_values, success, cov)
