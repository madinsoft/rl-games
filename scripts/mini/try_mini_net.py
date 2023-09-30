from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import array as A
from random import randint
from os import path
from tools.esvizu import EsVizu


# ____________________________________________________________ functions
def f(state):
    mini = min(state)
    target = [1 if s == mini else 0 for s in state]
    return target
    # res = [0, 0, 0, 0]
    # res[A(state).argmin()] = 1
    # return res


def random_values(size):
    inputs = []
    targets = []
    for i in range(size):
        state = [randint(1, 4) for j in range(4)]
        one_state = enc.state2ones(state)
        target = f(state)
        inputs.append(one_state)
        targets.append(target)
    return inputs, targets


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


def format_y(predictions):
    res = [i for i, value in enumerate(predictions) if value > .5]
    if len(res) == 0:
        maxi = max(predictions)
        res = [i for i, value in enumerate(predictions) if value == maxi]
    return res


# ____________________________________________________________ class
class KerasVizu(EsVizu):

    def __call__(self, epoch, logs):
        loss = logs['loss']
        accuracy = logs['accuracy']
        val_loss = logs['val_loss']
        val_accuracy = logs['val_accuracy']
        EsVizu.__call__(self, loss, accuracy, val_loss, val_accuracy)


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
model_path = '/home/patrick/projects/IA/my-2048/data2'
checkpoint_filepath = '/tmp/best'

enc = OneEncoder()
inputs, targets = all_values()
# size = 1000
# inputs, targets = random_values(size)
# print(inputs)
# print(targets)
# exit()

inputs = A(inputs)
targets = A(targets)

force = True

if force or not path.exists(f'{model_path}/saved_model.pb'):
    qmodel = Sequential()
    qmodel.add(Dense(units=4, kernel_initializer='uniform', activation='relu', input_dim=16))
    # qmodel.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    qmodel.add(Dense(units=4, kernel_initializer='uniform', activation='sigmoid'))
    qmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    viz = KerasVizu('loss', 'accuracy', 'val_loss', 'val_accuracy', dt=.16)
    plot_loss_callback = LambdaCallback(on_epoch_end=viz)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=0,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        # save_freq='epoch')
        save_best_only=True)

    stop_cb = EarlyStopping(monitor='loss', patience=1)

    # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=10, verbose=0, callbacks=[plot_loss_callback])
    # qmodel.fit(inputs, targets, batch_size=10, epochs=1000, verbose=2, callbacks=[plot_loss_callback, stop_cb])
    # qmodel.fit(inputs, targets, batch_size=32, epochs=1000, validation_data=(inputs, targets), verbose=0, callbacks=[plot_loss_callback, stop_cb, model_checkpoint_callback])
    qmodel.fit(inputs, targets, batch_size=32, epochs=1000, validation_split=.3, verbose=0, callbacks=[plot_loss_callback, stop_cb])
    # qmodel.fit(inputs, targets, batch_size=10, epochs=100, verbose=0, callbacks=[plot_loss_callback])
    qmodel.save(model_path)
else:
    qmodel = load_model(model_path)

# # size = 1000
# # inputs, targets = random_values(size)
# inputs, targets = all_values()
# predictions = qmodel.predict(inputs)

# count = 0
# for input_, target, value in zip(inputs, targets, predictions):
#     x = enc.ones2sate(input_)
#     y = format_y(target)
#     z = format_y(value)
#     if y != z:
#         print(x, y, z, value)
#         count += 1
# print(count)


# # qmodel = load_model(checkpoint_filepath)
# # size = 1000
# # inputs, targets = random_values(size)
# # predictions = qmodel.predict(inputs)

# # count = 0
# # for input_, target, value in zip(inputs, targets, predictions):
# #     x = enc.ones2sate(input_)
# #     y = format_y(target)
# #     z = format_y(value)
# #     if y != z:
# #         print(x, y, z, value)
# #         count += 1
# # print(count)
# # try:
# #     viz.freeze()
# # except:
# #     pass

