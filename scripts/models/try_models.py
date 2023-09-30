from utils import get_mini_policy
# from models.transformers import RangeTransfomer
from models.transformers import LabelTransformerFlat
from models.transformers import LabelTransformerOut
from models.basic_model import BasicModel
from models.tcbs import VizCB
from models.cbs import Evaluator
from models.cbs import Roller
from tools.sdi_vizu import SdiVizu
import gym
import gymini

inputs, targets = get_mini_policy()
for i in range(10):
    # model = BasicModel(
    #     nb_inputs=4,
    #     nb_targets=1,
    #     hidden_layers=[32, 16],
    #     learning_rate=.001,
    #     in_transformer=RangeTransfomer([1, 2, 3, 4]),
    #     out_transformer=RangeTransfomer([0, 1, 2, 3])
    # )
    model = BasicModel(
        nb_inputs=16,
        nb_targets=4,
        hidden_layers=[32, 16],
        learning_rate=.001,
        in_transformer=LabelTransformerFlat([1, 2, 3, 4]),
        out_transformer=LabelTransformerOut([0, 1, 2, 3])
    )

    viz = SdiVizu('wins', 'coverage', 'loss', dt=4, measurement='deepQ', model_name='mini')

    evaluator = Evaluator(inputs, targets, model)
    env = gym.make('mini-v0')
    roller = Roller(env, model, limit=20, nb=20)
    viz = VizCB(cbs=[roller, evaluator], viz=viz, objective=90., step=100)
    model.callback = viz

    model.learn(inputs, targets, epochs=20000)
    # print(model.history.history)
