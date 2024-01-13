from gym import Wrapper


class HistoryWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.state_history = []
        self.action_history = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.state_history.append(observation)
        self.action_history.append(action)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.state_history = []
        self.action_history = []
        return self.env.reset(**kwargs)

# # Usage
# env = HistoryWrapper(gym.make('CartPole-v1'))
# observation = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         observation = env.reset()
# print(env.state_history)
# print(env.action_history)
