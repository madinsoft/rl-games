import site
import gym
site.addsitedir('/home/patrick/pCloudDrive/docs/ML/rl-games/src')
import gymini
from players.player_mini import PlayerMini
from policies.policy_random import PolicyRandom


env = gym.make('mini-v0')
polic = PolicyRandom(env.actions)
player = PlayerMini(env, polic)
# player.run()
win = 0
lost = 0
for i in range(100):
    player.run()
    if player.reward == 10:
        win += 1

    else:
        lost += 1
    print('------------------------')
    print(player.length, player.init_state, player.state, player.reward)
    print('states =', player.states)
    print('actions =', player.actions)
    print('rewards =', player.rewards)
print('win', win)
print('lost', lost)
