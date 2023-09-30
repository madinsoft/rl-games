import site
from time import perf_counter
import gym
from numpy import array as A
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from players.player_mini import PlayerMini
from policies.policy_random import PolicyRandom
from policies.policy_ql import PolicyQL


# ________________________________________________________________
def evaluate(Q):
    count = 0
    for state, qvalues in pol.Q.items():
        if state == (4, 4, 4, 4):
            continue
        if qvalues.any():
            mini = min(state)
            goods = [i for i, v in enumerate(state) if v == mini]
            choice = qvalues.argmax()
            if choice not in goods:
                count += 1
        else:
            print(state, qvalues)
            count += 1
    return count


# for space in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
for space in [500]:
    env = gym.make('mini-v0')
    pol = PolicyQL()
    polic = PolicyRandom(env.actions)
    player = PlayerMini(env, polic, limit=200)
    win = 0
    lost = 0
    # print('start')
    start = perf_counter()
    for i in range(space):
        player.run()
        pol.learn(player.states, player.actions, player.rewards)
        if player.reward == 1:
            win += 1
        else:
            lost += 1
        # print('------------------------')
        # print(player.length, player.init_state, player.state, player.reward)
        # print('states =', player.states)
        # print('actions =', player.actions)
        # print('rewards =', player.rewards)
    elapsed = perf_counter() - start
    # print(f'elapsed = {elapsed:.2f} seconds')
    # print('win', win)
    # print('lost', lost)
    # print('------------------------')

    nb = len(pol.Q)

    count1 = 0
    for state, qvalues in pol.Q.items():
        if qvalues.any():
            mini = min(state)
            goods = [i for i, v in enumerate(state) if v == mini]
            choice = qvalues.argmax()
            # print(state, choice, goods, choice in goods, qvalues)

            if choice not in goods:
                count1 += 1
                print(state, choice, goods, choice in goods, qvalues)

    count2 = evaluate(pol.Q)
    print(f'{space} {count1} {count2} {nb} elapsed = {elapsed:.2f} seconds')


# pol = PolicyQL(.8, .9)
# states = [[2, 4, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3], [4, 3, 3, 3], [4, 4, 3, 3], [4, 3, 3, 3], [4, 4, 3, 3], [4, 4, 4, 3], [4, 4, 4, 4]]
# actions = [1, 0, 0, 1, 1, 1, 2, 3]
# rewards = [-1, 1, 1, 1, -1, 1, 1, 10]
# # states = [[2, 4, 3, 3], [3, 4, 3, 3], [4, 4, 3, 3], [4, 4, 4, 3], [4, 4, 3, 3], [4, 4, 4, 3], [4, 4, 4, 4]]
# # actions = [0, 0, 2, 2, 2, 3]

# # states = [[2, 4, 3, 3], [3, 4, 3, 3], [4, 4, 3, 3], [4, 4, 4, 3], [4, 4, 4, 4]]
# # actions = [0, 0, 2, 3]
# # rewards = [0, 0, 0, 10]

# for i in range(10):
#     pol.learn(states, actions, rewards)
#     print('------------------------')
#     for state, qvalues in pol.Q.items():
#         if qvalues.any():
#             print(state, qvalues)

# states = [[1, 3, 1, 2], [1, 3, 1, 1], [1, 2, 1, 1], [1, 2, 2, 1], [2, 2, 2, 1], [2, 1, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 1, 2], [1, 2, 1, 2], [2, 2, 1, 2], [2, 2, 2, 2], [2, 3, 2, 2], [2, 2, 2, 2], [2, 2, 2, 3], [2, 3, 2, 3], [2, 3, 2, 2], [2, 1, 2, 2], [0, 1, 2, 2]]
# actions = [3, 1, 2, 0, 1, 2, 1, 3, 0, 0, 2, 1, 1, 3, 1, 3, 1, 0]
# rewards = [-1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -2, -10]
# pol.learn(states, actions, rewards)
# print('------------------------')
# for state, qvalues in pol.Q.items():
#     if qvalues.any():
#         print(state, qvalues)

