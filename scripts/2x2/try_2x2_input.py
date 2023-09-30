import gym
import site
from numpy import array as A
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gy2048
# from gy2048.env import Base2048Env
# from numpy.random import choice

env = gym.make('2048-v0', width=2, height=2)

rewards = []
for i in range(1):
    done = False
    env.reset()
    # print(env.state)
    print('================================================================')
    env.render()
    print('================================================================')
    while not done:
        legal_actions = env.legal_actions
        action = int(input())
        board, reward, done, infos = env.step(action)
        # board, reward, done, infos = env.step(env.sample())
        # print(env.state, reward, done, env.nb_free_cases)
        print('================================================================')
        env.render()
        print('================================================================')

    rewards.append(reward)

print(rewards)
