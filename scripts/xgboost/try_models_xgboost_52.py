from policies.policy_ql import PolicyQL
from models.xgboost_model import XGModel
from numpy import array as A
from tools.sdi_vizu import SdiVizu
from models.cbs import Roller
from policies.best_close import Best2x2Policy
import gym
import gy2048


q_model = PolicyQL(gym.make('2048-v0', width=2, height=2), greedy=0, learning=.9, actualisation=.9)
best_model = Best2x2Policy()
xg_model = XGModel()

viz = SdiVizu('wins_xg', 'win_q', dt=4, measurement='xg', model_name='2x2')

objective = 95.
env = gym.make('2048-v0', width=2, height=2)
env.seed(1)
xg_roller = Roller(env, xg_model, nb=20)
q_roller = Roller(env, q_model, nb=20)

for i in range(10000):
    for j in range(100):
        done = False
        state = env.reset()
        states = [state]
        actions = []
        rewards = []
        while not done:
            # action = best_model(state)
            action = env.sample()
            actions.append(action)
            state, reward, done, infos = env.step(action)
            states.append(state)
            rewards.append(reward)

        q_model.learn(states, actions, rewards)
    q_states = A(list(q_model.Q.keys()))
    q_actions = A([qvalue.argmax() for qvalue in q_model.Q.values()])

    xg_model.learn(q_states, q_actions)
    wins_xg = xg_roller()
    winq = q_roller()
    viz(wins_xg, winq)
    print(i, wins_xg, winq)
    wins = min(wins_xg, winq)
    # if wins > objective:
    #     print('objective reached')
    #     break
"""
select mean(wins) as xg_wins,  mean(winsq) as q_wins from vizu..deepQ where $timeFilter group by time($__interval)
select mean(wins_xg) as xg_wins, mean(win_q) as q_wins, mean(wins_deep) as deep_wins, mean(cov_xg) as cov_xg, mean(cov_deep) as cov_deep, mean(cov_q) as cov_q from vizu..deepQ where $timeFilter group by time($__interval)
viz = SdiVizu('wins_xg', 'wins_deep', 'winq', dt=4, measurement='deepQ', model_name='mini')
"""
