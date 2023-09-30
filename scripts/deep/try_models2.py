from numpy import array as A
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from models.base_model import BasicModel
from models.base_model import Evaluator

inputs = A([(0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)])
# targets = A([.4, .2, .2, .4])
targets = A([1, 0, 0, 1])
# targets = A([.75, .25, .25, .75])

model = BasicModel(nb_inputs=4, nb_targets=1, hidden_layers=[16, 4], callback='sdi', metrics=['accuracy'])
net = model.net
eva = Evaluator(inputs, targets, model)
model.evaluator = eva
model.learn(inputs, targets)
