from time import perf_counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import array as A
# from random import randint
from os import path
from tools.esvizu import EsVizu


# ____________________________________________________________ functions
def proxima(values_possible, value):
    a = A(values_possible)
    b = a - value
    c = b**2
    return values_possible[c.argmin()]


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


def format_y(predictions):
    res = [i for i, value in enumerate(predictions) if value > .5]
    if len(res) == 0:
        maxi = max(predictions)
        res = [i for i, value in enumerate(predictions) if value == maxi]
    return res


# ____________________________________________________________ class
class KerasVizu(EsVizu):

    # def __call__(self, epoch, logs):
    #     loss = logs['loss']
    #     accuracy = logs['accuracy']
    #     val_loss = logs['val_loss']
    #     val_accuracy = logs['val_accuracy']
    #     EsVizu.__call__(self, loss, accuracy, val_loss, val_accuracy)

    def __call__(self, epoch, logs):
        loss = logs['loss']
        print(loss)
        # accuracy = logs['accuracy']
        # val_loss = logs['val_loss']
        # val_accuracy = logs['val_accuracy']
        EsVizu.__call__(self, loss)


# ==============================================================================
space = 5000
num_inputs = 4
num_hidden = 4
num_actions = 1
model_path = '/home/patrick/projects/IA/my-2048/data2'
checkpoint_filepath = '/tmp/best'

inputs, targets = all_values()
for i, t in zip(inputs, targets):
    print(i, t)

inputs = A(inputs)
targets = A(targets)

force = True
start = perf_counter()
if force or not path.exists(f'{model_path}/saved_model.pb'):
    qmodel = Sequential()
    qmodel.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=4))
    qmodel.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
    qmodel.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    qmodel.compile(optimizer='adam', loss='mean_squared_error')

    # viz = KerasVizu('loss', 'accuracy', 'val_loss', 'val_accuracy', dt=.16)
    viz = KerasVizu('loss', dt=.16)
    plot_loss_callback = LambdaCallback(on_epoch_end=viz)
    # model_checkpoint_callback = ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     verbose=0,
    #     save_weights_only=False,
    #     monitor='val_accuracy',
    #     mode='max',
    #     # save_freq='epoch')
    #     save_best_only=True)

    # stop_cb = EarlyStopping(monitor='loss', patience=1)

    qmodel.fit(inputs, targets, batch_size=100, epochs=space, verbose=0, callbacks=[plot_loss_callback])
    # qmodel.fit(inputs, targets, batch_size=100, epochs=space, verbose=0)
    qmodel.save(model_path)
else:
    qmodel = load_model(model_path)

possible_values = [.2, .4, .6, .8]
outs = qmodel.predict(inputs)
count = 0
for state, target, output in zip(inputs, targets, outs):
    choice = proxima(possible_values, output)
    print(state, target, choice, output)
    if choice != target:
        count += 1
elapsed = perf_counter() - start
print(f'count {count} elapsed {elapsed} seconds')
