from numpy import array as A
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from models.base_model import BasicModel
from models.base_model import VizCB
from models.base_model import Evaluator
from tools.sdi import Sdi

from time import perf_counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
from numpy import array as A
from random import randint
from os import path
from tools.esvizu import EsVizu
from tools.sdi_vizu import SdiVizu


# db = Sdi('vizu')
# db.drop_measurement('deepQ')


# inputs = A([(0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)])
# targets = A([(0, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)])
# # targets = A([1, 0, 0, 1])

# model = BasicModel(nb_inputs=4, nb_targets=4, hidden_layers=4, callback='sdi', metrics=['accuracy'])
# model.learn(inputs, targets)

# print(model(inputs[0]))
# print(model(inputs[1]))
# print(model(inputs[2]))
# print(model(inputs[3]))

def f(state):
    a = A(state)
    return float((a.argmin() + 1) / 5)


def all_values():
    inputs = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = A([i, j, k, h]) / 5
                    target = A([f(state)])
                    inputs.append(state)
                    targets.append(target)
    return inputs, targets


def proxima(values_possible, value):
    a = A(values_possible)
    b = a - value
    c = b**2
    action = c.argmin()
    return action, values_possible[action]


def evaluate(model):
    possible_values = [.2, .4, .6, .8]
    outs = net.predict(inputs)
    count = 0
    for state, output in zip(inputs, outs):
        action, value = proxima(possible_values, output)
        # print(state, target, choice, output)
        # if choice != target:
        # print(action, value, state[action], min(state))
        if state[action] == min(state):
            count += 1
    return count / len(inputs)


inputs, targets = all_values()
for i, t in zip(inputs, targets):
    print(i, t)

inputs = A(inputs)
targets = A(targets)

cho = 1

if cho == 1:
    model = BasicModel(nb_inputs=4, nb_targets=1, hidden_layers=[16, 4], callback='sdi', metrics=['accuracy'])
    net = model.net
    eva = Evaluator(inputs, targets, model)
    model.evaluator = eva
    model.learn(inputs, targets)
else:
    net = Sequential()
    net.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=4))
    net.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
    net.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    net.compile(optimizer='adam', loss='mean_squared_error')

    net.fit(inputs, targets, batch_size=100, epochs=5000, verbose=0, callbacks=[VizCB(evaluate, SdiVizu)])
