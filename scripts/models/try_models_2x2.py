from utils import get_best_2x2_policy
from models.transformers import RangeTransfomer
from models.basic_model import BasicModel
from models.cbs import VizCB
from models.cbs import Evaluator
from tools.sdi_vizu import SdiVizu
from models.cbs import Roller
import gym
import gy2048

inputs, targets = get_best_2x2_policy()
for i in range(10):
    model = BasicModel(
        nb_inputs=4,
        nb_targets=1,
        hidden_layers=[32, 16],
        # hidden_layers=[128, 16, 4],
        learning_rate=.0005,
        in_transformer=RangeTransfomer([1, 2, 3, 4]),
        out_transformer=RangeTransfomer([0, 1, 2, 3])
    )

    viz = SdiVizu('coverage', 'wins', 'loss', dt=4, measurement='deepQ', model_name='mini')

    evaluator = Evaluator(inputs, targets, model)
    env = gym.make('2048-v0', width=2, height=2)
    roller = Roller(env, model, nb=50)
    viz = VizCB(cbs=[evaluator, roller], viz=viz, objective=90., step=100)
    model.callback = viz

    model.learn(inputs, targets, epochs=20000)
    # print(model.history.history)
