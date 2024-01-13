import site
ROOT_DIR = '/home/patrick/pCloudDrive/docs/ML/rl-games'
site.addsitedir(f'{ROOT_DIR}')
site.addsitedir(f'{ROOT_DIR}/src/envs')
import gym
import gimini_env
from utils.gym_wrappers import HistoryWrapper

env = HistoryWrapper(gym.make('GiminiEnv-v0'))
state, options = env.reset(seed=0, options={})

for _ in range(100):
    action = env.action_space.sample()
    old_state = env.state.copy()
    state, reward, terminated, truncated, info = env.step(action)
    print(f'{old_state} {action} => {state}')
    # env.render()
    if truncated:
        print("lost !")
        break
    if terminated:
        print("won !")
        break
