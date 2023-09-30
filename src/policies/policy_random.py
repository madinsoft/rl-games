from policies.abstract_policy import AbstractPolicy


# ________________________________________________________________
class PolicyRandom(AbstractPolicy):

    def __init__(self, env):
        self.env = env

    def action(self, state):
        """ action to do from a specific policy """
        return self.env.sample()

    def save(self, path):
        pass

    def load(self, path):
        pass
