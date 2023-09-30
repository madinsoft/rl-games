import json
import gym
import site
from numpy import array as A
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gy2048
from policies.policy_close_best import PolicyCloseBest


# def choose_best(envi):
#     best = []
#     for action in envi.legal_actions:
#         board, reward, done, infos = env.explore(action)
#         close = envi.close(board)
#         best.append((reward + close / 10, action))

#     best.sort(key=lambda x: x[0])
#     return best[-1][1]


env = gym.make('2048-v0', width=2, height=2, cache=True)
best_model = PolicyCloseBest(env)

policy = {}
rewards = []
for i in range(1000):
    done = False
    state = env.reset()
    sum_reward = 0
    while not done:
        # action = choose_best(env)
        action = best_model(state)
        state, reward, done, infos = env.step(action)
        sum_reward += reward
        policy[''.join(str(i) for i in state)] = action
        # print(state, action, reward)
    print(sum_reward, env.objective_reached, len(env.cache))
    rewards.append(sum_reward)
    print(i, 'policy', len(policy))
    if len(policy) == 81:
        break

# print('policy', len(policy))
# with open('policy_states_2x2.json', 'w') as json_file:
#     json.dump(policy, json_file, indent=4)
