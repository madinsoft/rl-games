#!/usr/bin/env python
"""
Use MCTS with different custom policies
"""
from time import perf_counter
from numpy import array as A
from envs.game_2048 import Game2048
from policies.policy_mcts import PolicyMcts
from models.cbs import Roller
# from policies.policy_mixte import PolicyMaximizeRewardOnCorner
# from policies.policy_mixte import PolicyMaximizeSameNeighbors
# from policies.policy_mixte import PolicyMinimizeNbTile
# from policies.policy_mixte import PolicyMaximizeReward
from policies.policy_mixte import PolicyRandom


env = Game2048(size=3)
# for policy in [PolicyRandom, PolicyMaximizeReward, PolicyMinimizeNbTile, PolicyMaximizeSameNeighbors]:
for policy in [PolicyRandom]:
    pol = policy(env)
    print(pol.__class__.__name__)
    best_model = PolicyMcts(pol, env)
    for i in [20]:
        best_model.nb = i
        print(f'  Depth {i}')
        roller = Roller(env, best_model, nb=10, verbose=0)
        cronos = perf_counter()
        wins = roller()
        elapsed = perf_counter() - cronos
        max_states = A(roller.max_states)
        print(wins, roller.reward, max_states.max(), max_states.mean(), f'elapsed {elapsed:.2f} seconds')
        print(f'  wins: {wins}%, reward: {roller.reward}, max_states max: {max_states.max()}, max_states max: {max_states.mean()}, elapsed {elapsed:.2f} seconds')
