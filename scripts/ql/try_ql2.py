import site
from time import perf_counter
import gym
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from players.player_mini import PlayerMini
from policies.policy_ql import PolicyQL
from policies.policy_random import PolicyRandom
from tools.esplot import draw_curves
from random import seed
from tools.esvizu import EsVizu


# ________________________________________________________________
def percent(a, b):
    return (a / b * 10000) / 100


def coverage(Q):
    count = 0
    for state, qvalues in Q.items():
        if state == (4, 4, 4, 4):
            count += 1
        if qvalues.any():
            mini = min(state)
            goods = [i for i, v in enumerate(state) if v == mini]
            choice = qvalues.argmax()
            if choice in goods:
                count += 1
    return percent(count, len(Q))


def rollout(Q, nb):
    envi = gym.make('mini-v0')
    poli = PolicyQL(greedy=0)
    poli.Q = Q
    playeri = PlayerMini(envi, poli, limit=200)
    win = 0
    for i in range(nb):
        playeri.run()
        if playeri.reward == 1:
            win += 1
    return percent(win, nb)


# ================================
space = 500
roll_depth = 300
env = gym.make('mini-v0')
pol1 = PolicyQL(greedy=.1)
player1 = PlayerMini(env, pol1, limit=200)
pol9 = PolicyQL(greedy=.9)
player9 = PlayerMini(env, pol9, limit=200)
viz = EsVizu('cov1', 'cov9', 'win1', 'win9')
for i in range(space):
    player1.run()
    pol1.learn(player1.states, player1.actions, player1.rewards)
    player9.run()
    pol9.learn(player9.states, player9.actions, player9.rewards)
    viz(coverage(pol1.Q), coverage(pol9.Q), rollout(pol1.Q, roll_depth), rollout(pol9.Q, roll_depth))
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
# # pol.load('pol1.json')

# pol = PolicyQL(learning=.8, actualisation=.9, greedy=0)
# pol.load('pol9.json')
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
