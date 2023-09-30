import gym
# import json
from numpy import array as A
import site
from tensorflow.keras.callbacks import Callback
from tools.sdi_vizu import SdiVizu
# import tensorflow as tf
# from tensorflow import keras
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
from models.base_model import BasicModel
from utils import Roller
from utils import get_best_2x2_policy
from utils import Evaluator
from utils import RangeTransfomer
import gy2048


# ____________________________________________________________ Callbacks
class VizCB(Callback):
    """Callback that terminates training when  """

    def __init__(self, coverage, roller, step=100, objective=0.9):
        super().__init__()
        self.coverage = coverage
        self.roller = roller
        self.objective = objective
        self.viz = SdiVizu('loss', 'coverage', 'wins', 'reward', dt=4, measurement='deepQ', model_name='2048-2x2')
        # self.viz = SdiVizu('loss', 'coverage', dt=4, measurement='deepQ', model_name='2048-2x2')
        self.loss_history = []
        self.step = step
        self.wins = 0
        self.reward = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % (self.step * 10) == 0:
            self.wins, self.reward = roller(20)
            # self.loss_history = []
        if epoch % self.step == 0:
            success = self.coverage()
            # wins, mean_reward = roller(20)
            loss = logs.get('loss', 0.)
            self.loss_history.append(loss)

            if success >= self.objective:
                print(f'Epoch {epoch}: Reached objective, terminating training')
                self.model.stop_training = True
                self.model.save(f'last-good-model-2x2-{success}')

            if len(self.loss_history) > 100:
                tester = A(self.loss_history[-10:])
                rel = tester.std() / tester.mean()
                if rel < 1e-5:
                    print(f'Epoch {epoch}: No progress on error, stop here')
                    self.model.stop_training = True
                    self.model.save('bad-model-2x2')
            else:
                rel = 0
            success *= 100
            # wins *= 100
            # self.viz(loss, success)
            self.viz(loss, success, self.wins, self.reward)
            print(f'epoch {epoch}, loss {loss}, success {success}, wins {self.wins}, mean_reward {self.reward}, rel {rel}')
            # print(f'epoch {epoch}, loss {loss}, success {success}, rel {rel}')


# ============================================= main
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
    nbs.append((len(h['loss']), evaluator()))

print(nbs)

