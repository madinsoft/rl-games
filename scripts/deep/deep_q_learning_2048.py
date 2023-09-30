# from time import perf_counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tools.esvizu import EsVizu
from tools.sdi_vizu import SdiVizu
import gym
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gy2048


num_hidden = 32
show = 'influx'
width = 2
height = 2


# ________________________________________________________________ functions
def create_q_model():
    num_inputs = width * height
    num_actions = 4

    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="softmax")(common)
    return keras.Model(inputs=inputs, outputs=action)


def get_action(model, state):
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()
    # print('get action', state, action, action_probs)
    return action


def percent(a, b):
    return int(a / b * 10000) / 100


def rollout(model, nb):
    # start = perf_counter()
    mean_reward = 0
    for i in range(nb):
        # print(i, 'start rollout')
        envi.reset()
        state = envi.state
        done = False
        length = 0
        while not done:
            length += 1
            action = get_action(model, state)
            if not envi.is_action_legal(action):
                action = envi.sample()
            board, reward, done, _ = envi.step(action)
            state = envi.state
            # print(i, length, state, action, reward)
        mean_reward += reward
    mean_reward /= nb
    # elapsed = perf_counter() - start
    # print(f'{mean_reward}, {nb} elapsed {elapsed:.2f}s')
    return percent(mean_reward, nb)


# ===============================
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
envi = gym.make('2048-v0', width=width, height=height)
env.seed(seed)

num_actions = 4

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

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

if show == 'matplot':
    viz = EsVizu('wins', dt=2)
else:
    measurement = 'deepQ'
    viz = SdiVizu('wins', 'cov', dt=4, measurement=measurement, model_name=f'hid{num_hidden}')

loss, running_reward = 0, 0
objectif = 16
end = 1024
while True:  # Run until solved
    env.reset()
    if sum(env.state) != 2:
        continue
    state = np.array(env.state)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # if timestep % 10 == 0:
        #     wins = rollout(model, 20)
        #     viz(wins)
        #     print(episode_count, timestep, wins, episode_reward, running_reward)
        #     if wins > objectif:
        #         print(f'save deep_2048_{num_hidden}_{objectif}')
        #         model.save(f'/home/patrick/projects/IA/my-2048/data_models/deep_2048_{num_hidden}_{objectif}')
        #         objectif *= 2
        #         print(f'new objectif {objectif}')
        #     if wins > end:
        #         model.save(f'/home/patrick/projects/IA/my-2048/data_models/deep_2048_{num_hidden}')
        #         viz.freeze()
        #         exit()

        frame_count += 1

        # Use epsilon-greedy for exploration
        tirage = np.random.rand(1)[0]
        if frame_count < epsilon_random_frames or epsilon > tirage:
            # Take random action
            # action = np.random.choice(num_actions)
            action = env.sample()
            # print('random', epsilon, tirage, frame_count, epsilon_random_frames, frame_count < epsilon_random_frames, epsilon > tirage)
        else:
            # Predict action Q-values
            # From environment state
            action = get_action(model, state)
            # print('action', epsilon, tirage)

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        # state_next = np.array(state_next)
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
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % 10 == 0:
                wins = rollout(model, 100)
                cov = evaluate(model)
                viz(wins, cov)
                print(frame_count, timestep, episode_count, wins, cov, episode_reward, running_reward)
                if wins > objectif:
                    print(f'save deep_2048_{num_hidden}_{objectif}')
                    model.save(f'/home/patrick/projects/IA/my-2048/data_models/deep_2048_{num_hidden}_{objectif}')
                    objectif *= 2
                    print(f'new objectif {objectif}')
                if wins > end:
                    model.save(f'/home/patrick/projects/IA/my-2048/data_models/deep_2048_{num_hidden}')
                    viz.freeze()
                    exit()

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
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
