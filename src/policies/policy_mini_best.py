from numpy import array as A
from policies.abstract_policy import AbstractPolicy


# ________________________________________________________________
class PolicyBest(AbstractPolicy):

    def action(self, state):
        """ action to do from a specific policy """
        return A(state).argmin()

    def save(self, path):
        pass

    def load(self, path):
        pass
