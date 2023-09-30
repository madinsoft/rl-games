from gym.envs.registration import register
from .env import BaseMiniEnv

register(
    id='mini-v0',
    entry_point='gymini.env:BaseMiniEnv',
)
