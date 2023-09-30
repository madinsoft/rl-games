# from time import perf_counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tools.esvizu import EsVizu
from tools.sdi_vizu import SdiVizu
import gym
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
site.addsitedir('/home/patrick/projects/IA/my-2048/src/envs')
import gy2048
from utils import get_best_2x2_policy
from utils import Evaluator
from utils import Roller


num_hidden = 32
show = 'influx'
width = 2
height = 2

# ____________________________________________________________ BaseModel
class DeepModel:

    def __init__(self, nb_inputs, nb_targets, nb_hidden, **kwargs):
        inputs = layers.Input(shape=(nb_inputs,))
        common = layers.Dense(nb_hidden, activation="relu")(inputs)
        action = layers.Dense(nb_targets, activation="softmax")(common)
        self.net = keras.Model(inputs=inputs, outputs=action)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    def action(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.net(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action

    def __call__(self, states):
        if hasattr(states[0], '__iter__'):
            return self.net(states)
        return self.action(states)

    # _________________________________ expose interface of AioSdiEx
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return object.__getattribute__(self.net, attr)

# ___________________________________________ class
class Best2x2Policy:

    def __init__(self):
        self.env = gym.make('2048-v0', width=2, height=2)

    def __call__(self, state):
        self.env.state = state
        best = []
        for action in self.env.legal_actions:
            state, reward, done, infos = env.explore(action)
            close = self.env.close(state)
            best.append((reward + close / 10, action))

        best.sort(key=lambda x: x[0])
        return best[-1][1]


# ===============================
inputs, targets = get_best_2x2_policy()
evaluator = Evaluator(inputs, targets)

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# ________________________________________________________________ env
env = gym.make("2048-v0", width=width, height=height)  # Create the environment
env.seed(seed)

num_actions = 4

best_model = Best2x2Policy()
model = DeepModel(nb_inputs=4, nb_targets=4, nb_hidden=32)
model_target = DeepModel(nb_inputs=4, nb_targets=4, nb_hidden=32)

envi = gym.make('2048-v0', width=width, height=height)
roller = Roller(env, model)

# optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
# episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 500
# Number of frames for exploration
epsilon_greedy_frames = 10000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000
# Train the model after 4 actions
update_after_actions = 1
# How often to update the target network
update_target_network = 1000
# Using huber loss for stability
loss_function = keras.losses.Huber()

viz = SdiVizu('wins', 'reward', 'loss', dt=4, measurement='deepQ', model_name=f'hid{num_hidden}', clear=True)

loss, running_reward = 0, 0
while True:  # Run until solved
    env.reset()
    state = np.array(env.state)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        tirage = np.random.rand(1)[0]
        # if frame_count < epsilon_random_frames or epsilon > tirage:
        #     action = env.sample()
        # else:
        #     # action = get_action(model, state)
        #     action = model(state)

        action = best_model(state)

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        state_next, reward, done, _ = env.step(action)
        state_next = np.array(env.state)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.net.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                # q_values = model.net(state_sample)
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % 100 == 0:
                # wins = rollout(model, 100)
                success, mean_reward = roller(100)
                success *= 100
                # cov = evaluator(model)
                viz(success, mean_reward, loss)
                print(f'({frame_count}, {episode_count}, {timestep}) success: {success}, reward: {mean_reward}')
                if success > 80:
                    print(f'save deep_2048_{num_hidden}')
                    model.save(f'/home/patrick/projects/IA/my-2048/data_models/deep_2048_{num_hidden}_80')
                if success > 100:
                    model.save(f'/home/patrick/projects/IA/my-2048/data_models/deep_2048_{num_hidden}')
                    viz.freeze()
                    exit()

        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    episode_count += 1
