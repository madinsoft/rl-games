from numpy import array as A
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseModel


# ____________________________________________________________ BasicModel
class BasicModel(BaseModel):
    def __init__(self, nb_inputs, nb_targets, hidden_layers, **kwargs):
        self.net = self.build(nb_inputs, nb_targets, hidden_layers, **kwargs)
        self.history = None
        self.states = None
        self.targets = None
        # self.evaluator = kwargs.get('evaluator', None)
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
        model.add(Dense(units=nb_targets, kernel_initializer=kernel_initializer, activation='relu'))
        # model.add(Dense(units=nb_targets, kernel_initializer=kernel_initializer, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def learn(self, states, targets, **kwargs):
        callback = kwargs.get('callback', None)
        callback = callback or self.callback
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

        if callback:
            self.history = self.net.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[callback])
        else:
            self.history = self.net.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return self.history

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


# ===============================
if __name__ == '__main__':
    from utils import get_mini_policy
    from transformers import RangeTransfomer

    inputs, targets = get_mini_policy()
    model = BasicModel(
        nb_inputs=4,
        nb_targets=1,
        hidden_layers=[4],
        learning_rate=.0001,
        in_transformer=RangeTransfomer([1, 2, 3, 4]),
        out_transformer=RangeTransfomer([0, 1, 2, 3])
    )
    model.learn(inputs, targets, epochs=1000, verbose=1)
    print(model.history.history)
