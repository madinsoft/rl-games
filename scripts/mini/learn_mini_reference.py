import numpy as np
from numpy import array as A
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


# ________________________________________________________________ functions
def get_mini_policy():
    inputs = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    target = int(A(state).argmin())
                    inputs.append(state)
                    targets.append(target)

    inputs = A(inputs)
    targets = A(targets)
    return inputs, targets


# ____________________________________________________________ BaseTransformer
class BaseTransformer(ABC):

    # ________________________________________________ abstractmethod
    @abstractmethod
    def encode_value(self, value):
        pass

    @abstractmethod
    def decode_value(self, value):
        pass

    @abstractmethod
    def _round(self, values):
        pass

    @abstractmethod
    def length(self):
        pass

    # ________________________________________________ methods
    def encode(self, values):
        try:
            return self.encode_value(values)
        except TypeError:
            try:
                return self.encode_list(values)
            except TypeError:
                return self.encode_table(values)

    def __call__(self, values):
        return self.encode(values)

    def decode(self, values):
        try:
            return self.decode_table(values)
        except TypeError:
            try:
                return self.decode_list(values)
            except TypeError:
                return self.decode_value(values)

    def encode_list(self, values):
        return [self.encode_value(value) for value in values]

    def encode_table(self, table):
        return [self.encode_list(liste) for liste in table]

    def decode_list(self, values):
        return [self.decode_value(value) for value in values]

    def decode_table(self, table):
        return [self.decode_list(liste) for liste in table]


# ____________________________________________________________ Transformer
class LabelTransformer(BaseTransformer):

    def __init__(self, labels):
        self.table = {}
        self.inv_table = {}
        nb = len(labels)
        ones = np.zeros(nb)
        ones[-1] = 1
        self._length = 0
        for label in labels:
            ones = np.roll(ones, 1)
            tones = tuple(ones)
            self.table[label] = tones
            self.inv_table[tones] = label
            self._length = len(tones)

    # ________________________________________________ abstractmethod
    def encode_value(self, value):
        return self.table[value]

    def decode_value(self, value):
        try:
            return self.inv_table[self._round(value)]
        except KeyError:
            print('value', value)
            print('inv table', self.inv_table)
            raise

    def _round(self, values):
        i = A(values).argmax()
        ones = np.zeros(len(values))
        ones[i] = 1
        return tuple(ones)

    @property
    def length(self):
        return self._length

# ____________________________________________________________ Transformer
class LabelTransformerFlat(LabelTransformer):

    def encode_list(self, values):
        res = []
        for value in values:
            res += self.encode_value(value)
        return res


# ____________________________________________________________ Transformer
class LabelTransformerOut(LabelTransformer):

    def __call__(self, values):
        return values


# ____________________________________________________________ BasicModel
class BasicModel:
    def __init__(self, nb_inputs, nb_targets, hidden_layers, **kwargs):
        self.net = self.build(nb_inputs, nb_targets, hidden_layers, **kwargs)
        self.history = None
        self.states = None
        self.targets = None
        self.callback = kwargs.get('callback', None)
        self.in_transformer = kwargs.get('in_transformer', None)
        self.out_transformer = kwargs.get('out_transformer', None)

    def build(self, nb_inputs, nb_targets, hidden_layers, **kwargs):
        kernel_initializer = kwargs.get('kernel_initializer', 'uniform')
        learning_rate = kwargs.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=learning_rate)
        loss = kwargs.get('loss', 'mean_squared_error')
        metrics = kwargs.get('metrics', None)
        model = Sequential()
        try:
            first_layer = hidden_layers.pop(0)
        except AttributeError:
            first_layer = hidden_layers
            hidden_layers = []
        model.add(Dense(units=first_layer, kernel_initializer=kernel_initializer, activation='relu', input_dim=nb_inputs))
        for hidden_layer in hidden_layers:
            model.add(Dense(units=hidden_layer, kernel_initializer=kernel_initializer, activation='relu'))
        model.add(Dense(units=nb_targets, kernel_initializer=kernel_initializer, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def learn(self, states, targets, **kwargs):
        states = A(states)
        targets = A(targets)
        self.states = states
        self.targets = targets
        if self.in_transformer:
            states = A(self.in_transformer(states))
        if self.out_transformer:
            targets = A(self.out_transformer(targets))

        batch_size = kwargs.get('batch_size', 100)
        epochs  = kwargs.get('epochs', 5000)
        verbose = kwargs.get('verbose', 0)

        if self.callback:
            self.history = self.net.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[self.callback])
        else:
            self.history = self.net.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, states):
        states = A(states)
        if self.in_transformer:
            states = A(self.in_transformer(states))
        targets = self.net.predict(states)
        if self.out_transformer:
            targets = A(self.out_transformer.decode(targets))
        return targets

    def __call__(self, state):
        if self.in_transformer:
            state = A([self.in_transformer(state)])
        else:
            state = A([state])
        res = self.net.predict(state)
        res = res[0]
        if self.out_transformer:
            res = self.out_transformer.decode(res)
        try:
            return res[0]
        except TypeError:
            return res


# ____________________________________________________________ Callbacks
class StopCB(Callback):
    """Callback that terminates training when  """

    def __init__(self, objective=0.9, step=10):
        super().__init__()
        self.objective = objective
        self.loss_history = np.zeros(10)
        self.step = step
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            self.count += 1
            loss = logs.get('loss', 0.)
            self.loss_history[self.count % 10] = loss
            accuracy = logs.get('accuracy', 0.)

            if accuracy >= self.objective:
                print(f'Epoch {epoch}: Reached objective, terminating training')
                self.model.stop_training = True

            if epoch > 100:
                tester = self.loss_history
                rel = tester.std() / tester.mean()
                if rel < 1e-7:
                    print(f'Epoch {epoch}: No progress on error, stop here')
                    self.model.stop_training = True
            else:
                rel = 0
                tester = []
            print(f'epoch {epoch}, loss {loss}, success {accuracy}, rel {rel}')


# ===============================
if __name__ == '__main__':
    inputs, targets = get_mini_policy()
    model = BasicModel(
        nb_inputs=16,
        nb_targets=4,
        hidden_layers=[8, 4],
        learning_rate=.001,
        callback=StopCB(),
        metrics=['accuracy'],
        in_transformer=LabelTransformerFlat([1, 2, 3, 4]),
        out_transformer=LabelTransformerOut([0, 1, 2, 3])
    )
    model.learn(inputs, targets, epochs=20000)
