from gym.envs.registration import register
# from .env import GiminiEnv

register(
    id='GiminiEnv-v0',
    entry_point='gimini_env.env:GiminiEnv',
)

# from gym.envs.registration import register

# register(
#     id='GiminiEnv-v0',
#     entry_point='gimini_env:GiminiEnv',
# )
