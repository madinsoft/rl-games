from policies.policy_ql import PolicyQL
from utils import get_best_2x2_policy
from numpy import array as A
from models.transformers import LabelTransformer
from models.transformers import LabelTransformerOut
from models.transformers import LabelTransformerFlat
from models.basic_model import BasicModel
from models.cbs import VizCB
from models.cbs import Evaluator
from tools.sdi_vizu import SdiVizu
from models.cbs import Roller
from policies.best_close import Best2x2Policy
import gym
import gy2048


best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()
model1 = BasicModel(
    nb_inputs=20,
    nb_targets=4,
    hidden_layers=[128, 16, 4],
    learning_rate=.0005,
    in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
    out_transformer=LabelTransformerOut([0, 1, 2, 3])
)
model2 = BasicModel(
    nb_inputs=20,
    nb_targets=4,
    hidden_layers=[128, 16, 4],
    learning_rate=.0005,
    in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
    out_transformer=LabelTransformer([0, 1, 2, 3])
)
# viz1 = SdiVizu('win1', 'win2', 'coverage1', 'coverage2', 'loss', dt=4, measurement='compare', model_name='compare')
viz2 = SdiVizu('win1', 'win2', 'coverage1', 'coverage2', 'loss', dt=4, measurement='compare', model_name='compare')

objective = 90.
pol = PolicyQL(gym.make('2048-v0', width=2, height=2), greedy=0, learning=.9, actualisation=.9)
ev1 = Evaluator(inputs, targets, model1)
roller1 = Roller(gym.make('2048-v0', width=2, height=2), model1, nb=20)
ev2 = Evaluator(inputs, targets, model2)
roller2 = Roller(gym.make('2048-v0', width=2, height=2), model2, nb=20)
# vizu1 = VizCB(cbs=[roller1, roller2, ev1, ev2], viz=viz1, objective=objective, step=100)
vizu2 = VizCB(cbs=[roller2, roller1, ev2, ev1], viz=viz2, objective=objective, step=100)
# model1.callback = vizu1
model2.callback = vizu2

max_memory_length = 1000000

all_actions = []
all_states = []
env = gym.make('2048-v0', width=2, height=2)
for i in range(10000):
    print('game')
    for j in range(10):
        done = False
        state = env.reset()
        actions = []
        states = [state]
        rewards = []
        while not done:
            action = best_model(state)
            # action = env.sample()
            actions.append(action)
            all_actions.append(action)
            state, reward, done, infos = env.step(action)
            states.append(state)
            all_states.append(state)
            rewards.append(reward)
        pol.learn(states, actions, rewards)

    q_states = A(list(pol.Q.keys()))
    q_actions = A(list(pol.Q.values()))
    q_actions /= (q_actions.max() / .8)
    print('model1')
    model1.learn(q_states, q_actions, epochs=1000)
    print('model2')
    model2.learn(all_states, all_actions, epochs=1000)
    wins = roller1()
    if wins > objective:
        print('objective reached')
        break
    if len(all_states) > max_memory_length:
        break
