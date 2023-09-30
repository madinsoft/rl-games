import gym
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from gymini.evaluate_mini import evaluate
from policies.policy_random import PolicyRandom


env = gym.make('mini-v0')
polic = PolicyRandom(env.actions)
note = evaluate(polic)
print(note)
