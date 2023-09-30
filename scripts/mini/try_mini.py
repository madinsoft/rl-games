import site
import gym
from random import randint
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini


# ________________________________________________________________
class Player:

    def __init__(self, env):
        self.env = env
        self.actions = []
        self.rewards = []
        self.states = []

    def action_policy(self):
        """ action to do from policy """
        # actions = self.legal_actions
        actions = self.env.actions
        if actions:
            return actions[randint(0, len(actions) - 1)]
        return None

    @property
    def done(self):
        return self.env.is_done()

    def reset(self):
        self.env.reset()
        self.actions = []
        self.rewards = []
        self.states = [self.env.state.copy()]

    def run(self):
        self.reset()
        self.init_state = self.env.state.copy()
        done = False
        while not done:
            action = self.action_policy()
            done = self.step(action)
            # print(self.run_length, action, done, self.state, self.game_result, self.actions, self.states)
        # print(self.run_length, init_state, self.state, self.game_result)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # print('next_state', next_state, info, done)
        if info.get('valid', True):
            self.states.append(next_state)
            self.rewards.append(reward)
            self.actions.append(action)
        return done

    def is_action_legal(self, action):
        _, _, _, info = self.env.explore(action)
        return info.get('valid', True)

    @property
    def legal_actions(self):
        actions = []
        for action in self.env.actions:
            if self.is_action_legal(action):
                actions.append(action)
        return actions

    @property
    def game_result(self):
        if not self.env.is_done():
            return None
        if self.rewards:
            return self.rewards[-1]
        return 0

    @property
    def run_length(self):
        return len(self.states)

    @property
    def state(self):
        # return self.env.state
        return self.states[-1]


env = gym.make('mini-v0')
player = Player(env)
player.run()
print('states =', player.states)
print('actions =', player.actions)
print('rewards =', player.rewards)
# win = 0
# lost = 0
# for i in range(100):
#     player.run()
#     if player.game_result == 10:
#         win += 1
#         print(player.run_length, player.init_state, player.state, player.game_result)
#     else:
#         lost += 1
# print('win', win)
# print('lost', lost)
