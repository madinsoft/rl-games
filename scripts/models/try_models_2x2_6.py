from policies.policy_ql import PolicyQL
from utils import get_best_2x2_policy
from numpy import array as A
from models.transformers import LabelTransformerOut
from models.transformers import LabelTransformerFlat
from models.basic_model import BasicModel
from models.cbs import VizCB
from models.cbs import Evaluator
from tools.sdi_vizu import SdiVizu
from models.cbs import Roller
from policies.best_close import Best2x2Policy
import gym
import gy2048


best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()
model = BasicModel(
    nb_inputs=20,
    nb_targets=4,
    hidden_layers=[128, 16, 4],
    learning_rate=.0005,
    in_transformer=LabelTransformerFlat([0, 1, 2, 3, 4]),
    out_transformer=LabelTransformerOut([0, 1, 2, 3])
)

viz1 = SdiVizu('win_model', 'win_q', 'coverage_model', 'coverage_q', 'loss', dt=4, measurement='deepQ', model_name='better')
# viz2 = SdiVizu('qv', 'success', 'cov', dt=4, measurement='deepQ', model_name='2x2')

objective = 90.
evaluator = Evaluator(inputs, targets, model)
env = gym.make('2048-v0', width=2, height=2)
env1 = gym.make('2048-v0', width=2, height=2)
env2 = gym.make('2048-v0', width=2, height=2)
env3 = gym.make('2048-v0', width=2, height=2)
roller = Roller(env1, model, nb=20)
pol = PolicyQL(env2, greedy=0, learning=.9, actualisation=.9)
qeval = Evaluator(inputs, targets, pol)
qroll = Roller(env3, pol, nb=20)
viz = VizCB(cbs=[roller, qroll, evaluator, qeval], viz=viz1, objective=objective, step=100)
model.callback = viz

max_memory_length = 1000000

for i in range(10000):
    for j in range(10):
        done = False
        state = env.reset()
        actions = []
        states = [state]
        rewards = []
        while not done:
            action = best_model(state)
            # action = env.sample()
            actions.append(action)
            state, reward, done, infos = env.step(action)
            states.append(state)
            rewards.append(reward)
        pol.learn(states, actions, rewards)
        cov = qeval()
        success = qroll()
        size = len(pol.Q)
        q_values = pol.mean
        print(f'    Q {i} size {size} {q_values:.2f} coverage: {cov:.2f}% success: {success:.2f}%')
        # viz2(q_values, success, cov)

    q_states = A(list(pol.Q.keys()))
    q_actions = A(list(pol.Q.values()))
    q_actions /= (q_actions.max() / .8)
    # for state, action in zip(q_states, q_actions):
    #     print(state, action)
    model.learn(q_states, q_actions, epochs=1000)
    wins = roller()
    if wins > objective:
        print('objective reached')
        break
    if len(states) > max_memory_length:
        break
