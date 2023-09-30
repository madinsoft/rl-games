import numpy as np
from tensorflow.keras.callbacks import Callback


# ____________________________________________________________ Callbacks
class VizCB(Callback):
    """Callback that terminates training when  """

    def __init__(self, cbs=[], viz=None, objective=0.9, step=10):
        super().__init__()
        self.cbs = cbs
        self.objective = objective
        self.viz = viz
        self.loss_history = np.zeros(10)
        self.step = step
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            self.count += 1
            data_viz = []

            for cb in self.cbs:
                data_viz.append(cb(self.model))

            if 'accuracy' in logs:
                data_viz.append(logs['accuracy'])

            loss = logs.get('loss', 0.)
            self.loss_history[self.count % 10] = loss
            data_viz.append(loss)

            if self.viz:
                self.viz(*data_viz)

            if len(data_viz) > 1:
                success = data_viz[0]
                if success >= self.objective:
                    print(f'Epoch {epoch}: Reached objective, terminating training')
                    self.model.stop_training = True
                    self.model.save(f'last-good-model-2x2-{success}')
            else:
                success = 0

            if epoch > 100:
                tester = self.loss_history
                rel = tester.std() / tester.mean()
                if rel < 1e-7:
                    print(f'Epoch {epoch}: No progress on error, stop here')
                    self.model.stop_training = True
                    self.model.save('bad-model-2x2')
            else:
                rel = 0
                tester = []

            print(f'epoch {epoch}, loss {loss}, success {success}, rel {rel}')


# ____________________________________________________________ Callbacks
class StopCB(Callback):
    """Callback that terminates training when  """

    def __init__(self, cbs=[], viz=None, objective=0.9, step=10):
        super().__init__()
        self.objective = objective
        self.loss_history = np.zeros(10)
        self.step = step
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            self.count += 1
            loss = logs.get('loss', 0.)
            self.loss_history[self.count % 10] = loss
            accuracy = logs.get('accuracy', 0.)

            if accuracy >= self.objective:
                print(f'Epoch {epoch}: Reached objective, terminating training')
                self.model.stop_training = True

            if epoch > 100:
                tester = self.loss_history
                rel = tester.std() / tester.mean()
                if rel < 1e-7:
                    print(f'Epoch {epoch}: No progress on error, stop here')
                    self.model.stop_training = True
            else:
                rel = 0
                tester = []
            print(f'epoch {epoch}, loss {loss}, success {accuracy}, rel {rel}')
