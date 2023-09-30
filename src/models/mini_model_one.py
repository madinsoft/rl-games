import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from models.base_model import VizStop
from models.base_model import BaseModel


# ________________________________________________________________
class OneEncoder:

    def __init__(self, labels):
        self.table = {}
        self.inv_table = {}
        nb = len(labels)
        # ones = np.zeros(nb, dtype=int)
        ones = np.zeros(nb)
        ones[-1] = 1
        self.length = 0
        for label in labels:
            ones = np.roll(ones, 1)
            tones = tuple(ones)
            self.table[label] = tones
            self.inv_table[tones] = label
            self.length = len(tones)

    def encode(self, value):
        # print('encode', self.table, value)
        return self.table[value]

    __call__ = encode

    def decode(self, tones):
        i = tones.argmax()
        # ones = np.zeros(len(tones), dtype=int)
        ones = np.zeros(len(tones))
        ones[i] = 1
        tones = tuple(ones)
        return self.inv_table[tones]

    def state2ones(self, state):
        one_state = []
        for s in state:
            one_state += self.table[s]
        return one_state

    def ones2sate(self, one_state):
        state = []
        for i in range(4):
            ones = one_state[i * 4:(i + 1) * 4]
            state.append(self.inv_table[tuple(ones)])
        return state


# ________________________________________________________________
class MiniModelOne(BaseModel):

    def __init__(self, in_labels, nb_inputs, out_labels, nb_outputs, hidden_layers, debug=False, limit=100, kernel_initializer='uniform'):
        self.limit = limit
        self.incoder = OneEncoder(in_labels)
        self.outcoder = OneEncoder(out_labels)
        nb_state_ones = self.incoder.length * nb_inputs
        nb_output_ones = self.outcoder.length * nb_outputs
        # ________________________________________________________________
        net = Sequential()
        first_layer = hidden_layers.pop(0)

        net.add(Dense(units=first_layer, kernel_initializer=kernel_initializer, activation='relu', input_dim=nb_state_ones))
        for hidden_layer in hidden_layers:
            net.add(Dense(units=hidden_layer, kernel_initializer=kernel_initializer, activation='relu'))
        net.add(Dense(units=nb_output_ones, kernel_initializer=kernel_initializer, activation='tanh'))

        net.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.model = net
        self.debug = debug

    def __getitem__(self, state):
        return self.outcoder.decode(self(state))

    def __call__(self, state):
        resultats = self.model.predict([self.incoder.state2ones(state)])
        return resultats[0]

    def __setitem__(self, key, value):
        okey = self.coder(key)
        ovalue = self.coder(key)
        self.model.fit([okey], [ovalue], batch_size=32, epochs=10, verbose=0)

    def learn_actions(self, states, outputs):
        output_ones = [self.outcoder(output) for output in outputs]
        self.learn(states, output_ones)

    def learn(self, states, outputs):
        state_ones = [self.incoder.state2ones(state) for state in states]
        if self.debug:
            self.model.fit(state_ones, outputs, batch_size=5, epochs=self.limit, verbose=0, callbacks=[VizStop])
            plt.show()
        else:
            try:
                self.model.fit(state_ones, outputs, batch_size=5, epochs=self.limit, verbose=0)
            except Exception as e:
                print(e)
                print(state_ones)
                print(outputs)

    def predict(self, states):
        state_ones = [self.incoder.state2ones(tuple(state)) for state in states]
        output_ones = self.model.predict(state_ones)
        return output_ones

    def predict_actions(self, states):
        state_ones = [self.incoder.state2ones(tuple(state)) for state in states]
        output_ones = self.model.predict(state_ones)
        outputs = [self.outcoder.decode(output_one) for output_one in output_ones]
        return outputs

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model = load_model(model_path)

    @property
    def loss(self):
        return self.viz.loss

    @property
    def accuracy(self):
        return self.viz.accuracy
