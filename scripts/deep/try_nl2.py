from time import perf_counter
import gym
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from players.player_mini import PlayerMini
from policies.policy_nl import PolicyNL
from policies.policy_random import PolicyRandom
from policies.policy_ql import PolicyQL


# ________________________________________________________________
def percent(a, b):
    return (a / b * 10000) / 100


def rollout(pol, nb):
    envi = gym.make('mini-v0')
    poli = PolicyNL(greedy=0)
    poli.net = pol.net
    playeri = PlayerMini(envi, poli, limit=200)
    win = 0
    for i in range(nb):
        playeri.run()
        if playeri.reward == 10:
            win += 1
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


def test(pol):
    count = 0
    states = all_values()
    actions = pol.predict_actions(states)
    for state, action in zip(states, actions):
        # print(state, action, state[action], min(state), count, len(states))
        if state[action] != min(state):
            count += 1
    return count


# ________________________________________________________________
space = 1000
env = gym.make('mini-v0')
pol = PolicyNL(learning=.8)
polq = PolicyQL(learning=.8)
# states = [[2, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 4], [3, 4, 3, 4], [3, 3, 3, 4], [4, 3, 3, 4], [4, 4, 3, 4], [4, 2, 3, 4], [4, 2, 2, 4], [4, 2, 3, 4], [4, 3, 3, 4], [4, 3, 4, 4], [4, 4, 4, 4]]
# actions = [0, 3, 1, 1, 0, 1, 1, 2, 2, 1, 2, 1]
# rewards = [0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 1.0]
# pol.learn(states, actions, rewards)
# note = test(pol)
# print(note)

polic = PolicyRandom(env.actions)
player = PlayerMini(env, polic, limit=200)
win = 0
lost = 0
start = perf_counter()
for i in range(space):
    player.run()
    cronos = perf_counter()
    pol.learn(player.states, player.actions, player.rewards)
    polq.learn(player.states, player.actions, player.rewards)
    elapsed = perf_counter() - cronos
    note = test(pol)
    noteq = test(polq)
    if player.reward == 1:
        win += 1
    else:
        lost += 1

    print(i, note, noteq, len(player.states), player.reward, f'elapsed = {elapsed:.2f} seconds')
elapsed = perf_counter() - start
print(f'{space}  elapsed = {elapsed:.2f} seconds')

