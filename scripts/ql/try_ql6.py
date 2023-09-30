import site
from numpy import array as A
# from time import perf_counter
import gym
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gy2048
from tools.sdi_vizu import SdiVizu
from players.player_mini import PlayerMini
from policies.policy_ql import PolicyQL
from utils import get_best_2x2_policy
from utils import Roller


# ________________________________________________________________
def eval_action(env, action):
    state, reward, done, infos = env.explore(action)
    close = env.close(state)
    return reward + close / 10


# ___________________________________________ class
class Best2x2Policy:

    def __init__(self):
        self.env = gym.make('2048-v0', width=2, height=2)

    def __call__(self, state):
        self.env.state = state
        best = []
        for action in self.env.legal_actions:
            note = eval_action(self.env, action)
            best.append((note, action))
            # state, reward, done, infos = env.explore(action)
            # close = self.env.close(state)
            # best.append((reward + close / 10, action))

        best.sort(key=lambda x: x[0])
        return best[-1][1]


class Evaluator:

    def __init__(self, states, targets):
        self.states = states
        self.targets = targets

    def round(self, value):
        distances = (self.possible_values - value)**2
        closest = distances.argmin()
        return self.possible_values[closest]

    def __call__(self, model):
        outs = model.predict(self.states)
        count = 0
        for state, target, output in zip(self.states, self.targets, outs):
            # if output != -1:
            #     print(f'evaluator state {state} target {target} output {output}')
            if target == output:
                count += 1
            # try:
            #     value = A([self.round(i) for i in output])
            # except TypeError:
            #     value = A([self.round(output)])
            # if (value == target).all():
            #     count += 1
        return count / len(self.states)


# ================================
space = 1000
roll_depth = 300
env = gym.make('2048-v0', width=2, height=2)
env_pol = gym.make('2048-v0', width=2, height=2)
polQB = PolicyQL(env=env_pol, greedy=0)
done = False
length = 0

viz = SdiVizu('coverage', 'success', 'mean_reward', dt=4, measurement='deepQ', clear=True)

best_model = Best2x2Policy()

inputs, targets = get_best_2x2_policy()
evaluator = Evaluator(inputs, targets)

roller = Roller(env_pol, polQB)

for i in range(space):
    # print('')
    # print('================================================================')
    states = []
    actions = []
    rewards = []
    env.seed(i)
    # state = [1, 1, 0, 0]
    # env.state = state.copy()
    state = env.reset().copy()
    states.append(state)
    done = False
    while not done:
        # action = best_model(state)
        action = env.sample()
        next_state, reward, done, info = env.step(action)
        # print(f'state {state}, action {action}, next_state {next_state}, reward {reward}')
        actions.append(action)
        rewards.append(reward)
        state = next_state
        states.append(state)
    polQB.learn(states, actions, rewards)

    # print('--------------------------------------------------------')
    # for state, action, reward in zip(states, actions, rewards):
    #     print(state, action, reward)
    # print(states[-1])

    # print('--------------------------------------------------------')
    # for state, q_values in polQB.Q.items():
    #     try:
    #         action = polQB(state)
    #         print(f'state {state} action {action} q_values {q_values}')
    #     except ValueError:
    #         print(f'state {state} no action possible q_values {q_values}')

    # print('-------------------------------------------------------- test model with action list')
    for state, best_action in zip(states, actions):
        action = polQB(state)
        # print(f'state {state}, best action {best_action}, action {action}')
        if best_action != action:
            action_values = polQB.Q[tuple(state)]
            # print(f'Mismatch for state {state}')
            # print(f'action_values {action_values}, action {action}, best action {best_action}')
            note = eval_action(env, action)
            best_note = eval_action(env, best_action)
            if note != best_note:
                print(f'note {note}, best_note {best_note}')
                exit()
    # print('-------------------------------------------------------- eval')
    cov = evaluator(polQB) * 100
    success, mean_reward = roller(100)
    success *= 100
    print(f'{i} coverage {cov:.2f}% success {success:.2f}% mean_reward {mean_reward}')
    viz(cov, success, mean_reward)
    if success > 99:
        break

    # print('--------------------------------------------------------')
    # state = states[0]
    # env.seed(i)
    # env.state = state.copy()
    # done = False
    # while not done:
    #     action = polQB(state)
    #     action_values = polQB.Q[tuple(state)]
    #     best_action = best_model(state)
    #     next_state, reward, done, info = env.step(action)
    #     # note = eval_action(env, action)
    #     # best_note = eval_action(env, best_action)
    #     print(f'state {state}, best action {best_action}, action {action}, next_state {next_state}')
    #     if best_action != action:
    #         print(f'Mismatch for state {state}')
    #         print(f'action_values {action_values}, action {action}, best action {best_action}')
    #         note = eval_action(env, action)
    #         best_note = eval_action(env, best_action)
    #         if note != best_note:
    #             print(f'note {note}, best_note {best_note}')
    #             exit()
    #     state = next_state

polQB.save('/home/patrick/projects/IA/my-2048/data/q_2x2_2.json')
