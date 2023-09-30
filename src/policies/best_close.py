import gym


# ___________________________________________ class
class BestPolicy:

    def __init__(self, env):
        self.env = env

    def __call__(self, state):
        self.env.state = state
        best = []
        for action in self.env.legal_actions:
            state, reward, done, infos = self.env.explore(action, self.env.board)
            close = self.env.close(state)
            best.append((reward + close / 10, action))

        best.sort(key=lambda x: x[0])
        return best[-1][1]


# ___________________________________________ class
class Best2x2Policy(BestPolicy):

    def __init__(self):
        self.env = gym.make('2048-v0', width=2, height=2)

