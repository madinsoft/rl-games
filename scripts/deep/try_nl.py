import site
from random import randint
import numpy as np
from numpy import array as A
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from policies.policy_nl import OneEncoder
from policies.policy_nl import MiniModel
from tools.esvizu import EsVizu


# ____________________________________________________________ functions
def f(state):
    # mini = min(state)
    # target = [1 if s == mini else 0 for s in state]
    # return target
    a = A(state)
    return a.argmin()


def random_values(size):
    states = []
    targets = []
    for i in range(size):
        state = [randint(1, 4) for j in range(4)]
        target = f(state)
        states.append(state)
        targets.append(target)
    return states, targets


def all_values():
    states = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    target = f(state)
                    states.append(state)
                    targets.append(target)
    return states, targets


# ____________________________________________________________ class
class KerasVizu(EsVizu):

    def __call__(self, epoch, logs):
        loss = logs['loss']
        accuracy = logs['accuracy']
        val_loss = logs['val_loss']
        val_accuracy = logs['val_accuracy']
        EsVizu.__call__(self, loss, accuracy, val_loss, val_accuracy)


# =================================================================
in_labels = [1, 2, 3, 4]
out_labels = [0, 1, 2, 3]
model_path = 'test_model'
model = MiniModel(in_labels, 4, out_labels, 1, [4], debug=True, limit=500)

# size = 1000
# states, targets = random_values(size)
states, targets = all_values()
learn = True
if learn:
    model.learn_actions(states, targets)
    model.save(model_path)
    print(model.loss)
    print(model.accuracy)
else:
    model.load(model_path)

state = states[0]
print(state, model(state))
# count = 0
# for state, target in zip(states, targets):
#     output = model[state]
#     if target != output:
#         print(state, target, output)
#         count += 1
# print(count)
# count = 0
# outs = model.predict(states)
# for state, target, output in zip(states, targets, outs):
#     if target != output:
#         print(state, target, output)
#         count += 1
# print(count)

