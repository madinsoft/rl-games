import gym
import site
from numpy import array as A
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gy2048
# from gy2048.env import Base2048Env
# from numpy.random import choice

env = gym.make('2048-v0', width=2, height=2)

dico = {}
# rewards = []
for i in range(1000):
    done = False
    env.reset()
    print('state', env.state, sum(env.state))
    if sum(env.state) != 2:
        continue
    print('--------------------------------')
    print(env.state)

    while not done:
        # action = int(input())
        # board, reward, done, infos = env.step(action)
        state = tuple(env.state)
        action = env.sample()
        board, reward, done, infos = env.step(action)
        if state not in dico or dico[state]['reward'] < reward:
            dico[state] = {'reward': reward, 'action': action}

        print(env.state, reward, done, env.nb_free_cases)
        # env.render()
        # print('--------------------------------')

    # rewards.append(reward)

# print(rewards)
print(len(dico))
for state, data in dico.items():
    print(state, data)

dico = {
    (0, 1, 0, 1): 1,
    (0, 0, 1, 1): 0,
    (1, 1, 0, 0): 0,
    (1, 0, 1, 0): 1,
}
