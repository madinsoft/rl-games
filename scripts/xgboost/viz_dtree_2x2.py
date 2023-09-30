from time import perf_counter
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
from numpy import array as A

from envs.game_2048 import Game2048
from models.cbs import Runner
from policies.policy_mcts import PolicyMcts
from policies.policy_mixte import PolicyMaximizeSameNeighbors

cronos = perf_counter()
env = Game2048(size=2)
pol = PolicyMaximizeSameNeighbors(env)
best_mcts = PolicyMcts(pol, env, nb=5)

runner = Runner()
clf = DecisionTreeClassifier()
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
all_actions = []
all_states = []
for i in range(100):
    print(i, len(all_states), len(all_actions))
    max_best = runner(best_mcts, env)
    all_states += runner.states[:-1]
    all_actions += runner.actions

clf2 = clf.fit(all_states, all_actions)
viz = dtreeviz(clf2, A(all_states), A(all_actions), feature_names=['case 1', 'case 2', 'case 3', 'case 4'], class_names=['left', 'up', 'right', 'down'])
# viz.save('/home/patrick/projects/IA/my-2048/data/dtree.svg')
viz.view()
elapsed = perf_counter() - cronos
print(f'elapsed: {elapsed:.2f} seconds')
