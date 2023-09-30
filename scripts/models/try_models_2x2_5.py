from utils import get_best_2x2_policy
# from models.transformers import RangeTransfomer
from models.transformers import LabelTransformer
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
model = BasicModel(
    nb_inputs=20,
    nb_targets=4,
    hidden_layers=[8, 4],
    learning_rate=.0005,
    in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
    out_transformer=LabelTransformer([0, 1, 2, 3])
)

viz = SdiVizu('wins', 'coverage', 'loss', dt=4, measurement='deepQ', model_name='mini')

objective = 95.
evaluator = Evaluator(inputs, targets, model)
env = gym.make('2048-v0', width=2, height=2)
roller = Roller(env, model, nb=20)
viz = VizCB(cbs=[roller, evaluator], viz=viz, objective=objective, step=50)
model.callback = viz

actions = []
states = []
for i in range(10000):
    for j in range(10):
        done = False
        state = env.reset()
        while not done:
            action = best_model(state)
            actions.append(action)
            states.append(state)
            state, reward, done, infos = env.step(action)
        actions.append(action)
        states.append(state)
    model.learn(states, actions, epochs=500)
    wins = roller()
    if wins > objective:
        print('objective reached')
        break
# print(model.history.history)
