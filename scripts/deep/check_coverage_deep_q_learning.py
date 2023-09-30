from time import perf_counter
import tensorflow as tf
from tensorflow.keras.models import load_model
import gym
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gymini
from gymini.evaluate_mini import STATES, ACTIONS


def get_action(model, state):
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()
    return action


def percent(a, b):
    return int(a / b * 10000) / 100


def evaluate(model):
    count = 0
    outs = [get_action(model, state) for state in STATES]
    for action, output in zip(ACTIONS, outs):
        print(action, output)
        if action == output:
            count += 1
    return percent(count, len(STATES))


def evaluate2(model):
    count = 0
    for state in STATES:
        action = get_action(model, state)
        if state[action] == min(state):
            count += 1
        else:
            print(f'state {state}, action {action}, {state[action]}, {min(state)}')
    return percent(count, len(STATES))


def rollout(model, nb):
    start = perf_counter()
    limit = 20
    envi = gym.make('mini-v0')
    win = 0
    for i in range(nb):
        envi.reset()
        state = envi.state
        done = False
        length = 0
        while not done:
            length += 1
            action = get_action(model, state)
            state, reward, done, _ = envi.step(action)
            # print(i, length, state, action, reward)
            if length > limit:
                done = True
                break
        if reward == 1:
            win += 1
    elapsed = perf_counter() - start
    print(f'{win}, {nb} elapsed {elapsed:.2f}s')
    return percent(win, nb)


# ________________________________________________________________ model
choices = [8, 16, 32, 64, 128]
for choice in choices:
    model_path = f'/home/patrick/projects/IA/my-2048/data_models/deep{choice}'
    qmodel = load_model(model_path, compile=False)
    wins = rollout(qmodel, 100)
    note = evaluate2(qmodel)
    print(f'{choice}: note {note}% wins {wins}%')
