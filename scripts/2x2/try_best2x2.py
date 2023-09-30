"""
0.0 595.4 7 6.03
0.0 165.0 6 4.51
"""
from time import perf_counter
from numpy import array as A
from envs.game_2048 import Game2048
from policies.policy_mcts import PolicyMcts
from models.cbs import Roller

env = Game2048(size=3)
best_model = PolicyMcts(env, nb=100)
roller = Roller(env, best_model, nb=10, verbose=True)
cronos = perf_counter()
wins = roller()
elapsed = perf_counter() - cronos
max_states = A(roller.max_states)
print(wins, roller.reward, max_states.max(), max_states.mean(), f'elapsed {elapsed:.2f} seconds')
