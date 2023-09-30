from xgboost import XGBClassifier
from numpy import array as A

class XGModel:

    def __init__(self, **kwargs):
        self.net = XGBClassifier(**kwargs)

    def __call__(self, state):
        return self.net.predict(A([state]))[0]

    def learn(self, states, targets, **kwargs):
        return self.net.fit(states, targets, **kwargs)

    def load(self, file):
        raise NotImplementedError

    def save(self, file):
        self.net.save_model(file)
