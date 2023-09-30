import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from models.base_model import BasicModel


# ____________________________________________________________ BaseModel
class DeepModel(BasicModel):

    def __init__(self, nb_inputs, nb_targets, nb_hidden, **kwargs):
        optimizer = kwargs.pop('optimizer', keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0))
        loss = kwargs.pop('loss', keras.losses.Huber())

        super().__init__(nb_inputs, nb_targets, nb_hidden, optimizer=optimizer, loss=loss, **kwargs)
        self.net_target = self.build(nb_inputs, nb_targets, nb_hidden, optimizer=optimizer, loss=loss, **kwargs)
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []

    def build(self, nb_inputs, nb_targets, nb_hidden, **kwargs):
        inputs = layers.Input(shape=(nb_inputs,))
        common = layers.Dense(nb_hidden, activation='relu')(inputs)
        action = layers.Dense(nb_targets, activation='softmax')(common)
        return keras.Model(inputs=inputs, outputs=action)

    def learn(self, states, targets, **kwargs):
        running_reward = 0
        episode_count = 0
        frame_count = 0
        epsilon_random_frames = 500
        epsilon_greedy_frames = 10000.0
        update_after_actions = 4
        update_target_network = 1000
        
