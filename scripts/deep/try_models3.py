from numpy import array as A
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from models.base_model import BasicModel
from models.base_model import Evaluator
# from tools.sdi import Sdi

# db = Sdi('vizu')
# db.drop_measurement('deepQ')
inputs = A([(0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)])
targets = A([(.25, .75, .25, .25), (.75, .25, .25, .25), (.75, .25, .25, .25), (.25, .75, .25, .25)])

model = BasicModel(nb_inputs=4, nb_targets=4, hidden_layers=[16, 4], callback='sdi', metrics=['accuracy'])
net = model.net
eva = Evaluator(inputs, targets, model)
model.evaluator = eva
model.learn(inputs, targets)

