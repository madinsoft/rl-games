from players.abstract_player import AbtractPlayer


# ________________________________________________________________
class PlayerMini(AbtractPlayer):

    def __init__(self, env, policy, **kwargs):
        super().__init__(env, **kwargs)
        self._policy = policy

    def action(self):
        """ action to do from a specific policy """
        return self._policy.action(self.state)

    def after_run(self):
        # self._policy.learn(self.states, self.actions, self.rewards)
        pass

