from policies.policy_ql import PolicyQL
from numpy import array as A
from models.xgboost_model import XGModel
from models.transformers import LabelTransformer
from models.transformers import LabelTransformerOut
from models.transformers import LabelTransformerFlat
from models.basic_model import BasicModel
# from models.tcbs import VizCB
from tools.sdi_vizu import SdiVizu
from models.cbs import Roller
from policies.best_close import Best2x2Policy
from utils import get_best_2x2_policy
from models.cbs import Evaluator
import gym
import gy2048


q_model = PolicyQL(gym.make('2048-v0', width=2, height=2), greedy=0, learning=.9, actualisation=.9)
outer = LabelTransformer([0, 1, 2, 3])
best_model = Best2x2Policy()
xg_model = XGModel()
deep_model = BasicModel(
    nb_inputs=20,
    nb_targets=4,
    hidden_layers=[8, 4],
    learning_rate=.0005,
    in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
    out_transformer=LabelTransformerOut([0, 1, 2, 3])
)
viz = SdiVizu('wins_xg', 'wins_deep', 'win_q', 'cov_xg', 'cov_deep', 'cov_q', dt=4, measurement='deepQ', model_name='3x3')

objective = 95.
env = gym.make('2048-v0', width=2, height=2)
env.seed(1)
xg_roller = Roller(env, xg_model, nb=100)
deep_roller = Roller(env, deep_model, nb=100)
q_roller = Roller(env, q_model, nb=100)
inputs, targets = get_best_2x2_policy()
eval_xg = Evaluator(inputs, targets, xg_model)
eval_deep = Evaluator(inputs, targets, deep_model)
eval_q = Evaluator(inputs, targets, q_model)

all_actions = []
all_states = []
for i in range(10000):
    for j in range(10):
        done = False
        state = env.reset()
        states = [state]
        actions = []
        rewards = []
        while not done:
            all_states.append(state)
            # action = best_model(state)
            action = env.sample()
            all_actions.append(action)
            actions.append(action)
            state, reward, done, infos = env.step(action)
            states.append(state)
            rewards.append(reward)

        q_model.learn(states, actions, rewards)
    q_states = A(list(q_model.Q.keys()))
    q_states = A(list(q_model.Q.keys()))
    q_values = [outer._round(qvalue) for qvalue in q_model.Q.values()]
    q_actions = A([qvalue.argmax() for qvalue in q_model.Q.values()])

    xg_model.learn(q_states, q_actions)
    deep_model.learn(q_states, q_values, epochs=500)
    wins_xg = xg_roller()
    wins_deep = deep_roller()
    winq = q_roller()
    cov_xg = eval_xg()
    cov_deep = eval_deep()
    cov_q = eval_q()
    viz(wins_xg, wins_deep, winq, cov_xg, cov_deep, cov_q)
    print(i, wins_xg, wins_deep, winq, cov_xg, cov_deep, cov_q)
    wins = min(wins_xg, wins_deep, winq)
    if wins > objective:
        print('objective reached')
        break
"""
select mean(wins) as xg_wins,  mean(winsq) as q_wins from vizu..deepQ where $timeFilter group by time($__interval)
select mean(wins_xg) as xg_wins, mean(win_q) as q_wins, mean(wins_deep) as deep_wins, mean(cov_xg) as cov_xg, mean(cov_deep) as cov_deep, mean(cov_q) as cov_q from vizu..deepQ where $timeFilter group by time($__interval)
viz = SdiVizu('wins_xg', 'wins_deep', 'winq', dt=4, measurement='deepQ', model_name='mini')
"""
