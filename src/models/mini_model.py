from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from models.base_model import VizStop
from models.base_model import BaseModel


# ________________________________________________________________
class MiniModel(BaseModel):

    def __init__(self, nb_states, nb_actions, hidden_layers, debug=False, limit=100, kernel_initializer='uniform'):
        self.limit = limit
        # ________________________________________________________________
        net = Sequential()
        first_layer = hidden_layers.pop(0)
        net.add(Dense(units=first_layer, kernel_initializer=kernel_initializer, activation='relu', input_dim=nb_states))
        for hidden_layer in hidden_layers:
            net.add(Dense(units=hidden_layer, kernel_initializer=kernel_initializer, activation='relu'))
        net.add(Dense(units=nb_actions, kernel_initializer=kernel_initializer, activation='sigmoid'))
        net.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.model = net
        self.debug = debug
        self.history = None

    def __getitem__(self, state):
        resultats = self.model.predict([state])
        return resultats[0]

    def learn(self, states, actions):
        if self.debug:
            self.history = self.model.fit(states, actions, batch_size=5, epochs=self.limit, verbose=0, callbacks=[VizStop])
            plt.show()
        else:
            try:
                self.history = self.model.fit(states, actions, batch_size=5, epochs=self.limit, verbose=0)
            except Exception as e:
                print(e)
                print(states)
                print(actions)

    def predict(self, states):
        return self.model.predict(states)

    __call__ = predict

    @property
    def loss(self):
        return self.history.history['loss'][-1]

    @property
    def accuracy(self):
        return self.history.history['accuracy'][-1]


# ================================================================
if __name__ == '__main__':
    import gymini.evaluate_mini as em
    from gymini.evaluate_mini import evaluate

    in_labels = [1, 2, 3, 4]
    out_labels = [0, 1, 2, 3]
    states = em.STATES
    actions = em.TARGETS
    model = MiniModelOne(in_labels, 4, out_labels, 1, [4], debug=True, limit=200)
    model.learn(states, actions)
    outs = model(states)
    note = evaluate(model)
