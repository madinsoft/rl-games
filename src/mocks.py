from random import randint


# ____________________________________________________________ Mocks
def get_random_state():
    return [randint(0, 4) for i in range(4)]


def get_random_action():
    return randint(0, 3)


class ModelMock:
    def predict(self, inputs):
        return [[get_random_action()] for i in range(len(inputs))]

    def __call__(self, input):
        return randint(0, 3)


class EnvMock:
    def __init__(self, length=10):
        self.length = length
        self.count = 0

    def reset(self):
        self.count = 0

    def is_action_legal(self, action):
        return True

    def sample(self):
        return randint(0, 3)

    def step(self, action):
        self.count += 1
        done = self.count > self.length
        return get_random_state(), randint(0, 3), done, {'valid': True}

    @property
    def objective_reached(self):
        return randint(0, 4) == 2


