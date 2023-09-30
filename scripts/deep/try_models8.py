import gym
import json
from numpy import array as A
import site
from tools.sdi_vizu import SdiVizu
# import tensorflow as tf
# from tensorflow import keras
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
from models.base_model import BasicModel
from models.base_model import VizCB
from utils import Roller
from utils import get_best_2x2_policy
from utils import Evaluator
from utils import RangeTransfomer
import gy2048
# from utils import Evaluator


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

inputs, targets = get_best_2x2_policy()

env = gym.make('2048-v0', width=2, height=2)

model = BasicModel(
    nb_inputs=4,
    nb_targets=1,
    hidden_layers=[32, 16],
    learning_rate=.0001,
    metrics=['accuracy'],
    in_transformer=RangeTransfomer([0, 1, 2, 3, 4]),
    out_transformer=RangeTransfomer([0, 1, 2, 3])
)
evaluator = Evaluator(inputs, targets, model)
viz1 = SdiVizu('loss', 'coverage', dt=4, measurement='deepQ', model_name='2048-2x2', clear=True)
model.callback = VizCB(evaluate=evaluator, viz=viz1)
viz2 = SdiVizu('wins', 'reward', dt=4, measurement='deepQ', model_name='2048-2x2')

envi = gym.make('2048-v0', width=2, height=2)
roller = Roller(envi, model)
for i in range(10000):
    done = False
    state = env.reset()
    states = []
    actions = []
    while not done:
        states.append(state)
        action = best_model(state)
        actions.append(action)
        state, reward, done, infos = env.step(action)

    model.learn(states, actions, epochs=1000)
    wins, mean_reward = roller(20)
    viz2(wins * 100, mean_reward)
    print(f'wins {wins} mean_reward {mean_reward}')
