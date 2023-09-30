import gym
import json
from numpy import array as A
import site
from tensorflow.keras.callbacks import Callback
from tools.sdi_vizu import SdiVizu
# import tensorflow as tf
# from tensorflow import keras
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
from models.base_model import BasicModel
# from models.base_model import VizCB
from utils import Roller
from utils import get_best_2x2_policy
from utils import Evaluator
from utils import RangeTransfomer
import gy2048
# from utils import Evaluator


# ___________________________________________ class
class Best2x2Policy:

    def __init__(self):
        self.env = gym.make('2048-v0', width=2, height=2)

    def __call__(self, state):
        self.env.state = state
        best = []
        for action in self.env.legal_actions:
            state, reward, done, infos = env.explore(action)
            close = self.env.close(state)
            best.append((reward + close / 10, action))

        best.sort(key=lambda x: x[0])
        return best[-1][1]


# ____________________________________________________________ Callbacks
class VizCB(Callback):
    """Callback that terminates training when  """

    def __init__(self, coverage, roller, step=100, objective=0.9):
        super().__init__()
        self.coverage = coverage
        self.roller = roller
        self.objective = objective
        self.viz = SdiVizu('loss', 'coverage', 'wins', 'reward', dt=4, measurement='deepQ', model_name='2048-2x2')
        self.loss_history = []
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            success = self.coverage()
            wins, mean_reward = roller(100)
            loss = logs.get('loss', 0.)
            self.loss_history.append(loss)

            if success >= self.objective:
                print(f'Epoch {epoch}: Reached objective, terminating training')
                self.model.stop_training = True
                self.model.save(f'last-good-model-2x2-{success}')

            if epoch > 100:
                tester = A(self.loss_history[-10:])
                rel = tester.std() / tester.mean()
                if rel < 1e-7:
                    print(f'Epoch {epoch}: No progress on error, stop here')
                    self.model.stop_training = True
                    self.model.save('bad-model-2x2')
            else:
                rel = 0
            success *= 100
            wins *= 100
            self.viz(loss, success, wins, mean_reward)
            print(f'epoch {epoch}, loss {loss}, success {success}, wins {wins}, mean_reward {mean_reward}, rel {rel}')


# ============================================= main
best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()

nbs = []
for i in range(10):
    model = BasicModel(
        nb_inputs=4,
        nb_targets=1,
        hidden_layers=[32, 16],
        learning_rate=.0001,
        metrics=['accuracy'],
        in_transformer=RangeTransfomer([0, 1, 2, 3, 4]),
        out_transformer=RangeTransfomer([0, 1, 2, 3])
    )
    evaluator = Evaluator(inputs, targets, model)
    envi = gym.make('2048-v0', width=2, height=2)
    roller = Roller(envi, model)
    model.callback = VizCB(evaluator, roller)

    model.learn(inputs, targets, epochs=100000)
    h = model.history.history
    nbs.append((len(h['loss']), model.evaluate()))

print(nbs)

# states = []
# actions = []
# for i in range(1000):
#     done = False
#     state = env.reset()
#     while not done:
#         states.append(state)
#         action = best_model(state)
#         actions.append(action)
#         state, reward, done, infos = env.step(action)

# model.learn(states, actions, epochs=100000)
