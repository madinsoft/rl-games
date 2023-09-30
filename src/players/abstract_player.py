from abc import ABC, abstractmethod


# ________________________________________________________________
class AbtractPlayer(ABC):

    def __init__(self, env, limit=100):
        self.env = env
        self.init_state = None
        self.actions = None
        self.rewards = None
        self.states = None
        self.limit = limit
        self.reset()

    @abstractmethod
    def action(self):
        """ action to do from a specific policy """
        pass

    @abstractmethod
    def after_run(self):
        """ what to do after run """
        pass

    def reset(self):
        self.env.reset()
        self.init_state = self.env.state.copy()
        self.actions = []
        self.rewards = []
        self.states = [self.init_state]

    def run(self):
        # print('--------------------------------')
        self.reset()
        done = False
        while not done:
            action = self.action()
            done = self.step(action)
            if self.length > self.limit:
                return False
        self.after_run()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if info.get('valid', True):
            self.states.append(next_state)
            self.rewards.append(reward)
            self.actions.append(action)
            # print(self.length, next_state, reward, done, info)
        return done

    def is_action_legal(self, action):
        _, _, _, info = self.env.explore(action)
        return info.get('valid', True)

    @property
    def legal_actions(self):
        actions = []
        for action in self.env.ACTION_STRING:
            if self.is_action_legal(action):
                actions.append(action)
        return actions

    @property
    def done(self):
        return self.env.is_done()

    @property
    def length(self):
        return len(self.states)

    @property
    def state(self):
        return self.env.state
        # return self.states[-1]

    @property
    def reward(self):
        try:
            return self.rewards[-1]
        except IndexError:
            return 0

