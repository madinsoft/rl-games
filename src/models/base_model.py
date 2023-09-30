from numpy import array as A
from tensorflow.keras.models import load_model


# ____________________________________________________________ BaseModel
class BaseModel:

    def __init__(self, nb_inputs, nb_targets, hidden_layers, **kwargs):
        self.net = None
        self.history = None

    def build(self, nb_inputs, nb_targets, hidden_layers, **kwargs):
        pass

    def __getitem__(self, state):
        pass

    def __call__(self, state):
        res = self.net.predict(A([state]))[0]
        action = res.argmax()
        return action

    def __setitem__(self, key, value):
        pass

    def learn(self, states, targets, **kwargs):
        pass

    def predict(self, states):
        pass

    def predict_actions(self, states):
        pass

    def learn_actions(self, states, targets):
        pass

    def save(self, model_path):
        self.net.save(model_path)

    def load(self, model_path):
        self.net = load_model(model_path)

    def evaluate(self, inputs, targets):
        return self.net.evaluate(inputs, targets)

    @property
    def loss(self):
        return self.history.history['loss'][-1]

    @property
    def success(self):
        return self.history.history['success'][-1]
