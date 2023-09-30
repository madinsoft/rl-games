import site
# from time import perf_counter
import gym
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gymini
from players.player_mini import PlayerMini
from policies.policy_random import PolicyRandom
from policies.policy_ql import PolicyQL
from policies.policy_best import PolicyBest
from random import seed
# from tools.esvizu import EsVizu
from tools.sdi_vizu import SdiVizu


# ________________________________________________________________
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
    count = 0
    states = all_values()
    actions = pol.predict_actions(states)
    nb = len(states)
    for state, action in zip(states, actions):
        # print(state, action, count, len(states))
        if state[action] == min(state):
            count += 1
    return percent(count, nb)


# ================================
# seed(5)
space = 2000
roll_depth = 300
env = gym.make('mini-v0')
polQB = PolicyQL(greedy=0)
polQR = PolicyQL(greedy=0)
polB = PolicyBest()
polic = PolicyRandom(env)
playerB = PlayerMini(env, polB, limit=200)
playerR = PlayerMini(env, polic, limit=200)

# viz = EsVizu('covB', 'winB', 'covR', 'winR')
viz = SdiVizu('covB', 'winB', 'covR', 'winR', dt=4, measurement='deepQ')

for i in range(space):
    # print(i)
    playerB.run()
    polQB.learn(playerB.states, playerB.actions, playerB.rewards)
    playerR.run()
    polQR.learn(playerR.states, playerR.actions, playerR.rewards)
    if i % 10 == 0:
        viz(coverage(polQB), rollout(polQB, roll_depth), coverage(polQR), rollout(polQR, roll_depth))
viz.freeze()
