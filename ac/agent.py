import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import dtype

from ac.network import ActorCriticNetwork

tfd = tfp.distributions

class Agent:
    def __init__(self, env, lr: float=1e-4, gamma: float=0.99):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.gamma = gamma
        
        self.actions_space = [i for i in range(self.num_actions)]
        self.actor_critic_network = ActorCriticNetwork(num_actions=self.num_actions)
        self.optimizer = Adam(lr=lr)

    @tf.function
    def choose_action(self, observation):
        """Choose action by using actor network"""
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        _, actor_out = self.actor_critic_network(state)
        action_probabilities = tfd.Categorical(probs=actor_out)
        action = action_probabilities.sample() # sample action from actor net's output dist
        return tf.squeeze(action)
    
    @tf.function
    def learn(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, next_state: tf.Tensor, done: tf.Tensor):
        state = tf.convert_to_tensor([state], tf.float32)
        action = tf.convert_to_tensor(action, tf.int32)
        reward = tf.convert_to_tensor(reward, tf.float32)
        next_state = tf.convert_to_tensor([next_state], tf.float32)
        done = tf.cast(done, tf.float32)
        with tf.GradientTape() as tape:
            critic_out, actor_out = self.actor_critic_network(state)
            critic_out_nxt_state, _ = self.actor_critic_network(next_state)

            state_value = tf.squeeze(critic_out)
            nxt_state_value = tf.squeeze(critic_out_nxt_state)

            # calculate critic network error
            critic_err = reward + self.gamma * nxt_state_value * (1 - done) - state_value
            critic_loss = critic_err**2

            # calculate actor network error
            action_probs = tfd.Categorical(probs=actor_out)
            log_prob = tf.squeeze(action_probs.log_prob(action))
            actor_loss = -log_prob * critic_err

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.actor_critic_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic_network.trainable_variables))
        