import json
from numpy import array as A
import site
# import tensorflow as tf
# from tensorflow import keras
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from models.base_model import BasicModel
# from utils import Evaluator


with open('/home/patrick/projects/IA/my-2048/data/policy_states_2x2.json') as data_file:
    pol = json.load(data_file)

inputs = []
targets = []
for inupt_str, target in pol.items():
    inp = [float(i) / 5 for i in inupt_str]
    targ = target / 5
    inputs.append(inp)
    targets.append([targ])

inputs = A(inputs)
targets = A(targets)

nbs = []
for i in range(10):
    model = BasicModel(
        nb_inputs=4,
        nb_targets=1,
        hidden_layers=[32, 16],
        callback='sdi',
        learning_rate=.0001,
        metrics=['accuracy']
    )
    net = model.net
    # eva = Evaluator(inputs, targets)
    # model.evaluator = eva
    model.learn(inputs, targets, epochs=100000)
    # model.save('model-2x2')

    h = model.history.history
    nbs.append((len(h['loss']), model.evaluate()))

print(nbs)

"""
"""