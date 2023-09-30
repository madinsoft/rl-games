from numpy import array as A
import gym
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from players.player_mini import PlayerMini


# ________________________________________________________________
def f(state):
    # a = A(state)
    # nb = len(a) + 1
    # return float((a.argmin() + 1) / nb)
    a = A(state)
    return a.argmin()


def percent(a, b):
    return int(a / b * 10000) / 100


def rollout(pol, nb):
    envi = gym.make('mini-v0')
    playeri = PlayerMini(envi, pol, limit=200)
    win = 0
    for i in range(nb):
        playeri.run()
        if playeri.reward == 1:
            win += 1
    return percent(win, nb)


def all_states():
    states = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    states.append(state)
    return states


def evaluate(model):
    count = 0
    outs = model(STATES)
    for action, output in zip(ACTIONS, outs):
        if action == output:
            count += 1
    return percent(count, len(STATES))


# ________________________________________________________________
STATES = all_states()
ACTIONS = [f(state) for state in all_states()]
