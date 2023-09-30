from models.xgboost_model import XGModel
from numpy import array as A


class QXGModelOne(XGModel):

    def __init__(self, qpol, **kwargs):
        super().__init__(**kwargs)
        self.qpol = qpol
        # qpol.teacher = self

    def __call__(self, state):
        action = self.net.predict(A([state]))[0]
        if self.qpol.env.is_action_legal(action, state):
            return action
        return self.qpol.env.sample(state)

    def ones(self, state):
        res = self(state)
        return A(self.trans.encode(res))

    def update(self, **kwargs):
        q_states = A(list(self.qpol.Q.keys()))
        q_actions = A([qvalue.argmax() for qvalue in self.qpol.Q.values()])
        return self.learn(q_states, q_actions, **kwargs)


# ===============================
if __name__ == '__main__':
    from time import perf_counter
    from envs.game_2048 import Game2048
    from policies.policy_ql import PolicyQL
    from models.cbs import Roller

    env = Game2048(size=3)

    print('start load...')
    cronos = perf_counter()
    polQB = PolicyQL(env)
    polQB.load('/home/patrick/projects/IA/my-2048/data/q_3x3.json')
    elapsed = perf_counter() - cronos
    print(f'load {elapsed:.2f} seconds')

    print('start xg learn...')
    cronos = perf_counter()
    x_model = QXGModelOne(polQB)
    x_model.update()
    elapsed = perf_counter() - cronos
    x_model.save('xgboost_3x3.model')
    print(f'learn {elapsed:.2f} seconds')

    print('start perf...')
    xg_roller = Roller(env, x_model, nb=100)
    cronos = perf_counter()
    wins = xg_roller()
    elapsed = perf_counter() - cronos
    max_states = A(xg_roller.max_states)
    objective = 2**max_states.max()
    print(f'wins {wins}%, mean reward {xg_roller.reward}, max reached {objective}, mean state {max_states.mean()}, elapsed {elapsed:.2f} seconds')
