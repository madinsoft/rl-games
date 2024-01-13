
# ________________________________________________________________
class Player:

    def __init__(self, env, policy, limit=100):
        self._env = env
        self._policy = policy
        self._limit = limit

    def action(self):
        """ action to do from a specific policy """
        return self._policy.action(self._env.state)


    def run(self):
        # print('--------------------------------')
        self._env.reset()
        truncated = False
        terminated = False
        while not terminated and not truncated:
            action = self.action()
            _, _, terminated, truncated, _ = self._env.step(action)
            if self.length > self._limit:
                return False
        return True


