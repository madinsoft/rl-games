root = '/home/patrick/projects/IA/my-2048'
from sklearn.metrics import accuracy_score
from numpy import array as A
import json
import site
site.addsitedir(f'{root}/src')
from policies.policy_ql import PolicyQL
import gym
import gy2048


with open(f'{root}/scripts/policy_states_2x2.json') as data_file:
    goods = json.load(data_file)

inputs = []
targets = []
for inupt_str, target in goods.items():
    inp = [int(i) for i in inupt_str]
    targ = int(target)
    inputs.append(inp)
    targets.append(targ)

inputs = A(inputs)
targets = A(targets)


env = gym.make('2048-v0', width=2, height=2)
pol = PolicyQL()
pol.load(f'{root}/scripts/myq_2x2.pkl')

predictions = pol.predict(inputs)
accuracy = accuracy_score(targets, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

for target, prediction in zip(targets, predictions):
    print(target, prediction)
