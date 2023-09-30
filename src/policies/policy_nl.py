from random import random
from random import randint
from policies.abstract_policy import AbstractPolicy
from models.mini_model_one import MiniModelOne


# ________________________________________________________________
class PolicyNL(AbstractPolicy):

    def __init__(self, learning=.8, actualisation=.9, greedy=.2):
        self.learning = learning
        self.actualisation = actualisation
        self.greedy = greedy

        self.model = None
        self.reset()

    def reset(self):
        in_labels = [1, 2, 3, 4]
        out_labels = [0, 1, 2, 3]
        self.model = MiniModelOne(in_labels, 4, out_labels, 1, [4], debug=False, limit=800)

    def action(self, state):
        """ action to do from a specific policy """
        if self.greedy <= random():
            return self.model[state]
        else:
            return randint(0, 3)

    def learn(self, states, actions, rewards):
        net = self.model
        alpha = self.learning
        gamma = self.actualisation
        revstates = list(reversed(states[:-1]))
        revactions = list(reversed(actions))
        revrewards = list(reversed(rewards))
        next_state = tuple(states[-1])
        # inputs = []
        # targets = []
        cache_actions = {}
        for state, action, reward in zip(revstates, revactions, revrewards):
            state = tuple(state)
            # reward = 0
            try:
                if next_state in cache_actions:
                    next_actions = cache_actions[next_state]
                else:
                    next_actions = net(next_state)
                    cache_actions[next_state] = next_actions
                q_next = max(next_actions)
            except KeyError:
                q_next = 0

            if state in cache_actions:
                output = cache_actions[state]
            else:
                output = net(state)
                cache_actions[state] = output
            q = output[action]
            q_new = q + alpha * (reward + gamma * q_next - q)
            output[action] = q_new
            # print(state, action, output, 'q', q, 'q_new', q_new, q_next, reward)
            # inputs.append(state)
            # targets.append(output.tolist())
            # net.learn([state], [output.tolist()])
            next_state = state
        # net.learn(inputs, targets)
        inputs = list(cache_actions.keys())
        targets = [output.tolist() for output in cache_actions.values()]
        net.learn(inputs, targets)

        # outputs = net.predict(inputs)
        # print('--------------------------------')
        # for state, target, output in zip(inputs, targets, outputs):
        #     print(state, target, output)
        # print('--------------------------------')
        # print('stop here')
        # exit()

    def predict(self, states):
        return self.model.predict(states)

    def predict_actions(self, states):
        return self.model.predict_actions(states)

    def save(self, file_path_name):
        self.model.save(file_path_name)

    def load(self, file_path_name):
        self.model.load(file_path_name)

    learn_actions = learn

