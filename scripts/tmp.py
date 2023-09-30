from time import perf_counter

state = [1, 0, 0, 0, 0, 0, 0, 0, 1]
action = 0

# cronos = perf_counter()
# for i in range(1000):
#     state_str = ''.join(str(i) for i in state)
#     state_action_str = state_str + str(action)

#     dico = {}
#     dico[state_action_str] = 1
# elapsed1 = perf_counter() - cronos
# print(f'str elapsed = {elapsed1:.5f} seconds')

# cronos = perf_counter()
# for i in range(1000):
#     state_tuple = tuple(state)
#     state_action_tuple = (state_tuple, action)

#     dico = {}
#     dico[state_action_tuple] = 1
# elapsed2 = perf_counter() - cronos
# print(f'tuple elapsed = {elapsed2:.5f} seconds')
# ratio = elapsed1 / elapsed2
# print(f'ratio = {ratio}')

# LEFT = 0
# UP = 1
# RIGHT = 2
# DOWN = 3

# actions_to_str = {LEFT: 'left', UP: 'up', RIGHT: 'right', DOWN: 'down'}
# cronos = perf_counter()
# for i in range(1000):
#     for action in actions_to_str:
#         pass
# elapsed1 = perf_counter() - cronos
# print(f'tuple elapsed = {elapsed1:.5f} seconds')

# actions_to_str = {LEFT: 'left', UP: 'up', RIGHT: 'right', DOWN: 'down'}
# cronos = perf_counter()
# for i in range(1000):
#     for action in [0, 1, 2, 3]:
#         pass
# elapsed2 = perf_counter() - cronos
# print(f'tuple elapsed = {elapsed2:.5f} seconds')

# actions_to_str = {LEFT: 'left', UP: 'up', RIGHT: 'right', DOWN: 'down'}
# cronos = perf_counter()
# for i in range(1000):
#     for action in range(4):
#         pass
# elapsed2 = perf_counter() - cronos
# print(f'tuple elapsed = {elapsed2:.5f} seconds')

from numpy import array as A
import numpy as np

state = np.zeros(9)
locations = np.where(state == 0)
print(locations[0])
