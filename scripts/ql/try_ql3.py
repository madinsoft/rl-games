import site
# from time import perf_counter
import gym
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from players.player_mini import PlayerMini
from policies.policy_ql import PolicyQL
from policies.policy_random import PolicyRandom
from policies.policy_nl import PolicyNL
from policies.policy_nl import OneEncoder
from tools.esplot import draw_curves
from random import seed
from tools.esvizu import EsVizu


# ________________________________________________________________
def percent(a, b):
    return (a / b * 10000) / 100


# def coverage(Q):
#     count = 0
#     for state, qvalues in Q.items():
#         if state == (4, 4, 4, 4):
#             count += 1
#         if qvalues.any():
#             mini = min(state)
#             goods = [i for i, v in enumerate(state) if v == mini]
#             choice = qvalues.argmax()
#             if choice in goods:
#                 count += 1
#     return percent(count, len(Q))


def rollout(pol, nb):
    print('rollout')
    envi = gym.make('mini-v0')
    greedy = pol.greedy
    pol.greedy = 0
    playeri = PlayerMini(envi, pol, limit=200)
    win = 0
    for i in range(nb):
        playeri.run()
        if playeri.reward == 1:
            win += 1
    pol.greedy = greedy
    return percent(win, nb)


def all_values():
    states = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    states.append(state)
    return states


def coverage(pol):
    print('coverage')
    count = 0
    states = all_values()
    actions = pol.predict_actions(states)

    for state, action in zip(states, actions):
        # print(state, action, count, len(states))
        if state[action] != min(state):
            count += 1
    return count


# ================================
out_labels = [0, 1, 2, 3]
a = OneEncoder(out_labels)

space = 500
roll_depth = 300
env = gym.make('mini-v0')
polNL = PolicyNL(greedy=.1)
player1 = PlayerMini(env, polNL, limit=200)
polQL = PolicyQL(greedy=.1)
player9 = PlayerMini(env, polQL, limit=200)
viz = EsVizu('covNL', 'covQL', 'winNL', 'winQL')
for i in range(space):
    print(i)
    player1.run()
    polNL.learn(player1.states, player1.actions, player1.rewards)
    player9.run()
    polQL.learn(player9.states, player9.actions, player9.rewards)
    viz(coverage(polNL), coverage(polQL), rollout(polNL, roll_depth), rollout(polQL, roll_depth))
viz.freeze()

# space = 5000
# coverages = []
# wins = []
# cov_labels = []
# wins_labels = []
# # for greedy in [.1, .2, .5, .8, .9, 1]:
# for greedy in [.1, .9]:
#     seed(1)
#     print('------------------------', greedy)
#     env = gym.make('mini-v0')
#     pol = PolicyQL(learning=.8, actualisation=.9, greedy=greedy)
#     player = PlayerMini(env, pol, limit=200)
#     curve_coverage = []
#     curve_rollout = []
#     for i in range(space):
#         player.run()
#         pol.learn(player.states, player.actions, player.rewards)
#         cove = coverage(pol.Q)
#         curve_coverage.append(cove)
#         if cove == 100:
#             break
#         p = rollout(pol.Q, 1000)
#         curve_rollout.append(p)
#         print(f'  {i} cove {cove}% win {p}%')

#     coverages.append(curve_coverage)
#     wins.append(curve_rollout)
#     cov_labels.append(f'cov {greedy}')
#     wins_labels.append(f'win {greedy}')

#     # curves.append(curve)
#     # labels.append(greedy)
#     # count = evaluate(pol.Q)
#     # elapsed = perf_counter() - start
#     # print(f'greedy {greedy} space: {i} count: {count} won {win} {lost} elapsed = {elapsed:.2f} seconds')
#     # name = int(greedy * 10)
#     # pol.save(f'pol{name}.json')
# draw_curves(coverages + wins, cov_labels + wins_labels)

# # greedy = .1
# # pol = PolicyQL(learning=.8, actualisation=.9, greedy=greedy)
# # pol.load('polNL.json')

# pol = PolicyQL(learning=.8, actualisation=.9, greedy=0)
# pol.load('polQL.json')
# seed(1)
# space = 10
# env = gym.make('mini-v0')
# note = evaluate(pol.Q)

# player = PlayerMini(env, pol, limit=200)
# win = 0
# lost = 0
# start = perf_counter()
# for i in range(space):
#     player.run()
#     # pol.learn(player.states, player.actions, player.rewards)
#     print('----------------')
#     print(player.reward)
#     for state, action in zip(player.states, player.actions):
#         mini = min(state)
#         goods = [i for i, v in enumerate(state) if v == mini]
#         print(state, action, goods)
#     if player.reward == 10:
#         win += 1
#     else:
#         lost += 1
#     print(f'space: {i} note: {note} won {win} {lost}')
