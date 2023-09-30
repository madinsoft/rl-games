from policies.abstract_policy import AbstractPolicy


# ________________________________________________________________
class PolicyRL(AbstractPolicy):

    def __init__(self, env, policy):
        super().__init__(env)
        self._policy = policy

    def action(self, state):
        """ action to do from a specific policy """
        return self._policy.action(self.state)

    def learn(self, actions, states, rewards):
        pass
