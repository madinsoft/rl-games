from utils import get_best_2x2_policy
from models.xgboost_model import XGModel
from models.cbs import Roller
from models.cbs import Evaluator
import gym
import gy2048

inputs, targets = get_best_2x2_policy()

inputs = inputs.tolist()
targets = targets.tolist()

print(inputs[:5], type(inputs))
print(targets[:5], type(targets))

model = XGModel()
model.fit(inputs, targets)

objective = 95.
env = gym.make('2048-v0', width=2, height=2)
env.seed(1)
roller = Roller(env, model, nb=20)
evaluator = Evaluator(inputs, targets, model)
note = evaluator()
wins = roller(20)
print(note, wins)

