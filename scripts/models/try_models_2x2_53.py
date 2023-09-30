from policies.policy_ql import PolicyQL
from numpy import array as A
from utils import get_best_2x2_policy
from models.transformers import RangeTransfomer
from models.transformers import LabelTransformer
from models.transformers import LabelTransformerOut
from models.transformers import LabelTransformerFlat
from models.basic_model import BasicModel
from models.tcbs import VizCB
from models.cbs import Evaluator
from tools.sdi_vizu import SdiVizu
from models.cbs import Roller
# from policies.best_close import Best2x2Policy
import gym
import gymini


pol = PolicyQL(gym.make('mini-v0'), greedy=0, learning=.9, actualisation=.9)
outer = LabelTransformer([0, 1, 2, 3])
# best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()
# model = BasicModel(
#     nb_inputs=20,
#     nb_targets=4,
#     hidden_layers=[8, 4],
#     learning_rate=.0005,
#     in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
#     out_transformer=LabelTransformer([0, 1, 2, 3])
# )
model = BasicModel(
    nb_inputs=4,
    nb_targets=1,
    hidden_layers=[8, 4],
    learning_rate=.0005,
    in_transformer=RangeTransfomer([0, 1, 2, 3, 4]),
    out_transformer=RangeTransfomer([0, 1, 2, 3])
)
model1 = BasicModel(
    nb_inputs=20,
    nb_targets=4,
    hidden_layers=[8, 4],
    learning_rate=.0005,
    in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
    out_transformer=LabelTransformerOut([0, 1, 2, 3])
)
viz = SdiVizu('wins1', 'coverage', 'wins2', 'winsq', 'loss', dt=4, measurement='deepQ', model_name='mini')

objective = 95.
evaluator = Evaluator(inputs, targets, model)
evaluator1 = Evaluator(inputs, targets, model1)
env = gym.make('mini-v0')
env.seed(1)
roller = Roller(env, model, nb=20, limit=20)
roller1 = Roller(env, model1, nb=20, limit=20)
rollerq = Roller(env, pol, nb=20, limit=20)
# viz = VizCB(cbs=[roller, evaluator, roller1, rollerq], viz=viz, objective=objective, step=100)
# model.callback = viz
viz = VizCB(cbs=[roller1, evaluator1, roller, rollerq], viz=viz, objective=objective, step=100)
model1.callback = viz

all_actions = []
all_states = []
for i in range(10000):
    print('play')
    for j in range(10):
        done = False
        state = env.reset()
        states = [state]
        actions = []
        rewards = []
        limit = 0
        while not done:
            all_states.append(state)
            # action = best_model(state)
            action = env.sample()
            all_actions.append(action)
            actions.append(action)
            state, reward, done, infos = env.step(action)
            if limit > 20:
                done = True
                reward = -10
            states.append(state)
            rewards.append(reward)
            limit += 1
            # print(f'state {state} action {action}')

        pol.learn(states, actions, rewards)
    q_states = A(list(pol.Q.keys()))
    # q_actions = A(list(pol.Q.values()))
    # q_actions /= (q_actions.max() / .8)
    q_actions = [outer._round(qvalue) for qvalue in pol.Q.values()]

    print('model2')
    # model.learn(all_states, all_actions, epochs=500)
    model.learn(q_states.copy(), q_actions.copy(), epochs=500)
    print('model1')
    model1.learn(q_states.copy(), q_actions.copy(), epochs=500)
    wins = roller1()
    if wins > objective:
        print('objective reached')
        break
# print(model.history.history)
