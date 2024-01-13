# ________________________________________________________________
class PolicyRandom:

    def __init__(self, env):
        self._env = env

    def action(self, state):
        """ action to do from a specific policy """
        return self._env.sample_legal_action(state)

