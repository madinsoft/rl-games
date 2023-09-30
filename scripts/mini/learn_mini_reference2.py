from time import perf_counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from numpy import array as A


# ____________________________________________________________ functions
def f(state):
    mini = min(state)
    target = [1 if s == mini else 0 for s in state]
    return target


def all_values():
    inputs = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    one_state = enc.state2ones(state)
                    target = f(state)
                    inputs.append(one_state)
                    targets.append(target)
    return inputs, targets


# ____________________________________________________________ class
class OneEncoder:

    table = {
        1: [1, 0, 0, 0],
        2: [0, 1, 0, 0],
        3: [0, 0, 1, 0],
        4: [0, 0, 0, 1],
    }

    inv_table = {
        (1, 0, 0, 0): 1,
        (0, 1, 0, 0): 2,
        (0, 0, 1, 0): 3,
        (0, 0, 0, 1): 4,
    }

    def encode(self, value):
        return OneEncoder.table[value]

    def decode(self, value):
        return OneEncoder.inv_table[tuple(value)]

    def state2ones(self, state):
        one_state = []
        for s in state:
            one_state += OneEncoder.table[s]
        return one_state

    def ones2sate(self, one_state):
        state = []
        for i in range(4):
            ones = one_state[i * 4:(i + 1) * 4]
            state.append(OneEncoder.inv_table[tuple(ones)])
        return state


# ==============================================================================
num_inputs = 4
num_hidden = 4
num_actions = 1

enc = OneEncoder()
inputs, targets = all_values()

inputs = A(inputs)
targets = A(targets)

qmodel = Sequential()
qmodel.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=16))
qmodel.add(Dense(units=4, kernel_initializer='uniform', activation='sigmoid'))
qmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
cronos = perf_counter()
hist = qmodel.fit(inputs, targets, batch_size=32, epochs=1000, verbose=0)
elapsed = perf_counter() - cronos
print(f'elapsed: {elapsed:.2f} seconds')

import matplotlib.pylab as plt
histo = hist.history
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.plot(histo["loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.plot(histo["accuracy"])
plt.show()
