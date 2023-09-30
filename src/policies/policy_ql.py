import json
import pickle
import numpy as np
from numpy import array as A
from numpy.random import choice
from policies.abstract_policy import AbstractPolicy


# ________________________________________________________________
class PolicyQL(AbstractPolicy):

    def __init__(self, env, greedy=0, learning=.8, actualisation=.9, teacher=None):
        super().__init__(env, greedy=greedy)
        self.Q = self.reset()
        self.learning = learning
        self.actualisation = actualisation
        self.action_space_dimension = 4
        self.teacher = teacher

    def reset(self):
        self.Q = {}
        return self.Q

    def _action(self, state):
        """ action to do from a specific policy """
        Q = self.Q
        state = tuple(state)
        try:
            action_values = Q[state]
        except KeyError:
            if self.teacher:
                Q[state] = self.teacher.ones(state)
            else:
                Q[state] = np.zeros(self.action_space_dimension)
            action_values = Q[state]
            legal_actions = self.env.legal_actions(state)
            if len(legal_actions) > 0:
                return choice(legal_actions)
            else:
                return 0

        legal_actions = self.env.legal_actions(state)
        action_values = A([value if i in legal_actions else -1 for i, value in enumerate(self.Q[state])])
        return action_values.argmax()

    @property
    def total(self):
        somme = 0
        for qvalues in self.Q.values():
            somme += qvalues.sum()
        return somme

    @property
    def mean(self):
        return self.total / len(self.Q)

    def learn(self, states, actions, rewards):
        Q = self.Q
        alpha = self.learning
        gamma = self.actualisation
        revstates = list(reversed(states[:-1]))
        revactions = list(reversed(actions))
        revrewards = list(reversed(rewards))
        next_state = tuple(states[-1])
        for state, action, reward in zip(revstates, revactions, revrewards):
            state = tuple(state)
            try:
                q = Q[state][action]
            except KeyError:
                if self.teacher:
                    Q[state] = self.teacher.ones(state)
                    q = Q[state][action]
                else:
                    Q[state] = np.zeros(self.action_space_dimension)
                    q = 0

            try:
                Q[state][action] = q + alpha * (reward + gamma * Q[next_state].max() - q)
            except KeyError:
                if self.teacher:
                    Q[next_state] = self.teacher.ones(state)
                    Q[state][action] = q + alpha * (reward + gamma * Q[next_state].max() - q)
                else:
                    Q[state][action] = q + alpha * (reward - q)
            next_state = state

    learn_actions = learn

    def predict(self, states):
        actions = []
        for state in states:
            try:
                actions.append(int(self.Q[tuple(state)].argmax()))
            except KeyError:
                actions.append(-1)
        return actions

    predict_actions = predict

    def save(self, file_path_name):
        if file_path_name.endswith('.pkl'):
            pickle.dump(self.Q, open(file_path_name, 'wb'))
        elif file_path_name.endswith('.json'):
            Q = {}
            for state, actions in self.Q.items():
                Q[''.join(str(i) for i in state)] = list(actions)
            with open(file_path_name, 'w') as json_file:
                json.dump(Q, json_file, indent=4)
        else:
            raise TypeError(f'Unknown format {format} to save model')

    def load(self, file_path_name):
        if file_path_name.endswith('.pkl'):
            self.Q = pickle.load(open(file_path_name, 'rb'))
        elif file_path_name.endswith('.json'):
            with open(file_path_name) as data_file:
                Q = json.load(data_file)
                self.Q = {}
                for state, actions in Q.items():
                    state = tuple(int(e) for e in state)
                    actions = A(actions)
                    self.Q[state] = actions
        else:
            raise TypeError(f'Unknown format {format} to load model')


# ===============================
if __name__ == '__main__':
    from time import perf_counter
    from envs.game_2048 import Game2048
    from models.cbs import Roller
    from policies.policy_mcts import PolicyMcts
    from policies.policy_mixte import PolicyMaximizeSameNeighbors

    env = Game2048(size=3)
    pol = PolicyMaximizeSameNeighbors(env)
    best_model = PolicyMcts(pol, env, nb=5)
    qpol = PolicyQL(env)
    rollerq = Roller(env, qpol, nb=10, verbose=3)
    print(f'start q learning ')
    for j in range(10):
        done = False
        state = env.reset()
        states = [state]
        actions = []
        rewards = []
        while not done:
            action = best_model(state)
            actions.append(action)
            state, reward, done, infos = env.step(action)
            states.append(state)
            rewards.append(reward)
            # print(max(state), state, reward)

        qpol.learn(states, actions, rewards)
        print(f'  {j}: max state', max(state))

    cronos = perf_counter()
    wins = rollerq()
    elapsed = perf_counter() - cronos
    max_states = A(rollerq.max_states)
    print(wins, rollerq.reward, max_states.max(), max_states.mean(), f'elapsed {elapsed:.2f} seconds')
