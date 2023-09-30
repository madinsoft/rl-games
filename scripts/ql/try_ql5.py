import site
from numpy import array as A
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
limit = 200
space = 5
roll_depth = 300
env = gym.make('mini-v0')
polQB = PolicyQL(greedy=0)

done = False
length = 0

# state = [1, 1, 1, 1]
# env.state = state.copy()

for i in range(space):
    print('')
    print('================================================================')
    states = []
    actions = []
    rewards = []
    state = env.reset().copy()
    states.append(state)
    done = False
    while not done:
        action = A(state).argmin()
        next_state, reward, done, info = env.step(action)
        print(f'state {state}, action {action}, next_state {next_state}')
        actions.append(action)
        rewards.append(reward)
        state = next_state
        states.append(state)
        if length > limit:
            done = True
    polQB.learn(states, actions, rewards)

    print('--------------------------------------------------------')
    for state, action, reward in zip(states, actions, rewards):
        print(state, action, reward)
    print(states[-1])

    print('--------------------------------------------------------')
    for state, q_values in polQB.Q.items():
        action = polQB(state)
        print(f'state {state} action {action} q_values {q_values}')

    print('--------------------------------------------------------')
    state = states[0]
    env.state = state.copy()
    done = False
    while not done:
        action = polQB(state)
        action_values = polQB.Q[tuple(state)]
        # action = action_values.argmax()
        best_action = A(state).argmin()
        next_state, reward, done, info = env.step(action)
        print(f'state {state}, best action {best_action}, action {action}, next_state {next_state}')
        if best_action != action:
            print(f'action_values {action_values}, action {action}, best action {best_action}')
            exit()
        state = next_state
        if length > limit:
            done = True

