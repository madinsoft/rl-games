from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from time import perf_counter
from numpy import array as A
from tools.esvizu import EsVizu
from os import path


# ____________________________________________________________ functions
def percent(a, b):
    return int(a / b * 10000) / 100


def proxima(values_possible, value):
    a = A(values_possible)
    b = a - value
    c = b**2
    return values_possible[c.argmin()]


def f(state):
    a = A(state)
    nb = len(a) + 1
    return float((a.argmin() + 1) / nb)


def all_values():
    inputs = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            state = A([i, j]) / 5
            target = f(state)
            inputs.append(state)
            targets.append(target)
    return inputs, targets


def evaluate(inputs, targets, outs):
    count = 0
    possible_values = [0.3333333333333333, 0.6666666666666666]
    # possible_values = [.2, .4, .6, .8]
    for state, target, output in zip(inputs, targets, outs):
        choice = proxima(possible_values, output)
        # print(state, target, choice, output)
        if choice == target:
            count += 1
    nb = len(inputs)
    return percent(count, nb)


# ____________________________________________________________ class
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline """

    def __init__(self, baseline=0.9):
        super().__init__()
        self.baseline = baseline
        self.viz = EsVizu('loss', 'accuracy', dt=.16)

    def on_epoch_end(self, epoch, logs=None):
        outs = qmodel.predict(inputs)
        accuracy = evaluate(inputs, targets, outs)
        loss = logs.get('loss', 0)
        if loss > 0:
            while loss < 10:
                loss *= 10
        self.viz(loss, accuracy)
        print(epoch, loss, accuracy)
        if accuracy >= self.baseline:
            print(f'Epoch {epoch}: Reached baseline, terminating training')
            self.model.stop_training = True


# ==============================================================================
space = 5000
model_path = '/home/patrick/projects/IA/my-2048/model_mini'

inputs, targets = all_values()
inputs = A(inputs)
targets = A(targets)

force = False
start = perf_counter()
# if force or not path.exists(f'{model_path}/saved_model.pb'):
#     plot_loss_callback = TerminateOnBaseline(baseline=99)
#     qmodel = Sequential()
#     qmodel.add(Dense(units=1, kernel_initializer='uniform', activation='relu', input_dim=2)
#     qmodel.compile(optimizer='adam', loss='mean_squared_error')
#     qmodel.fit(inputs, targets, batch_size=100, epochs=space, verbose=0, callbacks=[plot_loss_callback])
#     qmodel.save(model_path)
# else:
#     qmodel = load_model(model_path)
qmodel = load_model(model_path)
outs = qmodel.predict(inputs)
accuracy = evaluate(inputs, targets, outs)
print(accuracy)

elapsed = perf_counter() - start
print(f'elapsed {elapsed} seconds')
