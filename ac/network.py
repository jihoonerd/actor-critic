from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense


class ActorCriticNetwork(tf.keras.Model):

    def __init__(self, num_actions: int, fc1_dims: int = 1024, fc2_dims: int = 512):
        super().__init__()
        self.num_actions = num_actions

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')

        self.critic = Dense(1, activation=None)
        self.actor = Dense(num_actions, activation='softmax')

    @tf.function
    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.fc1(state)
        common = self.fc2(x)
        critic_out = self.critic(common)
        actor_out = self.actor(common)
        return critic_out, actor_out
