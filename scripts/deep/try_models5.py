from numpy import array as A
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from models.base_model import BasicModel
from models.base_model import Evaluator

inputs = A([(0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)])
# targets = A([(0, 1), (1, 0), (1, 0), (0, 1)])
# targets = A([(0, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)])
targets = A([(.4, .6, .4, .4), (.6, .4, .4, .4), (.6, .4, .4, .4), (.4, .6, .4, .4)])

nbs = []
for i in range(10):
    model = BasicModel(
        nb_inputs=4,
        nb_targets=4,
        hidden_layers=[16, 4],
        callback='stop',
        learning_rate=.0005,
        epochs=5000,
        metrics=['accuracy']
    )
    # model = BasicModel(nb_inputs=4, nb_targets=4, hidden_layers=[16, 4], metrics=['accuracy'])
    # model = BasicModel(nb_inputs=4, nb_targets=4, hidden_layers=[16, 4])
    net = model.net
    eva = Evaluator(inputs, targets, model)
    model.evaluator = eva
    model.learn(inputs, targets)

    h = model.history.history
    nbs.append((len(h['loss']), model.evaluate()))

print(nbs)

"""
.01 [(31, 1.0), (191, 0.5), (201, 0.0), (191, 0.5), (201, 0.0), (201, 0.0), (201, 0.0), (31, 1.0), (201, 0.0), (31, 1.0)]
.0001 [1141, 3631, 1211, 2491, 3061, 2431, 2481, (1761), (1831), 311]
.0005 [(2131, 0.5), (2511, 0.5), (1031, 1.0), (11, 1.0), (2871, 0.25), (31, 1.0), (781, 1.0), (521, 1.0), (791, 1.0), (791, 1.0)]
.0005 [(41, 1.0), (41, 1.0), (2031, 0.5), (481, 0.25), (141, 1.0), (381, 1.0), (261, 1.0), (2031, 0.5), (471, 0.0), (601, 0.5)]
"""