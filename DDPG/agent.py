# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:agent.py
# software: PyCharm


import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf


class ReplayBuffer:

    def __init__(self, max_size, state_shape, action_shape):
        self.mem_size = max_size
        self.cntr = 0

        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size,), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size,), dtype=np.bool)

    def store_transition(self, state, reward, action, next_state, done):
        index = self.cntr % self.mem_size
        # store transition
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.done_memory[index] = done

        self.cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        done = self.done_memory[batch]

        return state, reward, action, next_state, done


class Actor(keras.Model):

    def __init__(self, dim1, dim2):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(dim1, activation='relu')
        self.dense2 = layers.Dense(dim2, activation=None)
        self.tanh = keras.activations.tanh

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)  # (batch, action_shape)
        y = self.tanh(x)  # (-1, 1)

        return y


class Critic(keras.Model):

    def __init__(self, dim1, dim2):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(dim1, activation='relu')
        self.dense2 = layers.Dense(dim2, activation=None)

    def call(self, inputs):
        state = inputs[0]
        action = inputs[1]
        inputs_ = layers.Concatenate()([state, action])
        x = self.dense1(inputs_)
        y = self.dense2(x)  # (batch,)
        y = tf.squeeze(y, axis=1)

        return y


def soft_updating(model1, model2, tau):
    length = len(model1.get_weights())
    new_weights = [(1 - tau) * model2.get_weights()[i] + tau * model1.get_weights()[i] for i in range(length)]
    model2.set_weights(new_weights)


class Agent:

    def __init__(self, state_shape, action_shape,
                 batch_size, actor_lr, critic_lr,
                 actor_dim1, actor_dim2, critic_dim1, critic_dim2,
                 max_size=1000000, gamma=0.99, tau=0.01):
        self.replay_buffer = ReplayBuffer(max_size=max_size,
                                          state_shape=state_shape,
                                          action_shape=action_shape)
        self.actor_fix = Actor(dim1=actor_dim1, dim2=actor_dim2)
        self.actor_change = Actor(dim1=actor_dim1, dim2=actor_dim2)
        # self.actor_fix.compile(optimizer=Adam(learning_rate),
        #                        loss=lambda y_true, y_pred: -self.critic_change([y_true, y_pred]))
        # self.actor_change.compile(optimizer=Adam(learning_rate),
        #                           loss=lambda y_true, y_pred: -self.critic_change([y_true, y_pred]))
        self.critic_fix = Critic(dim1=critic_dim1, dim2=critic_dim2)
        self.critic_change = Critic(dim1=critic_dim1, dim2=critic_dim2)
        # self.critic_fix.compile(optimizer=Adam(critic_lr), loss='mean_squared_error')
        # self.critic_change.compile(optimizer=Adam(critic_lr), loss='mean_squared_error')

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.optimizer = keras.optimizers.Adam(actor_lr)
        self.optimizer2 = keras.optimizers.Adam(critic_lr)
        self.initiate()

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store_transition(state, reward, action, next_state, done)

    def initiate(self):
        """synchronize parameters when instance agent class"""
        inputs1 = [keras.Input(shape=(4,)), keras.Input(shape=(1,))]
        self.critic_change(inputs1)
        self.critic_fix(inputs1)
        inputs2 = keras.Input(shape=(4,))
        self.actor_fix(inputs2)
        self.actor_change(inputs2)
        self.actor_fix.set_weights(self.actor_change.get_weights())
        self.critic_fix.set_weights(self.critic_change.get_weights())

    def learn(self):
        if self.replay_buffer.cntr < self.batch_size:
            return

        # start learning
        # sample from replay buffer
        # 1.train actor with tf api
        state, reward, action, next_state, done = self.replay_buffer.sample_buffer(batch_size=self.batch_size)

        with tf.GradientTape() as tape:
            actor_action = self.actor_change(state)  # (batch, 1)
            actor_loss = tf.reduce_mean(-self.critic_change([state, actor_action]))
        gradients = tape.gradient(actor_loss, self.actor_change.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor_change.trainable_variables))

        # 2.train critic with keras api
        done_int = done.astype('float32')
        # ##########################################
        # reward = np.expand_dims(reward, axis=1)  ## wrong code
        # ##########################################
        # self.critic_change.train_on_batch([state, action], Q_target)
        with tf.GradientTape() as tape:
            next_action = self.actor_fix(next_state)
            Q_target = reward + (1.0 - done_int) * self.gamma * self.critic_fix([next_state, next_action])
            # tf.stop_gradient(Q_target)
            Q_value = self.critic_change([state, action])
            critic_loss = tf.losses.mean_squared_error(Q_target, Q_value)
        critic_grad = tape.gradient(critic_loss, self.critic_change.trainable_variables)
        self.optimizer2.apply_gradients(zip(critic_grad, self.critic_change.trainable_variables))

        # TODO: soft updating used in DDPG
        # print(self.actor_fix.get_weights())
        # print(self.actor_change.get_weights())
        # self.actor_fix.set_weights(self.actor_change.get_weights())
        # self.critic_fix.set_weights(self.critic_change.get_weights())
        soft_updating(self.actor_change, self.actor_fix, self.tau)
        soft_updating(self.critic_change, self.critic_fix, self.tau)


if __name__ == '__main__':
    critic_ = Critic(dim1=100, dim2=1)
    inputs_ = [keras.Input(shape=(10,)), keras.Input(shape=(4,))]
    critic_(inputs_)
    # critic_.summary()
    print(critic_.get_weights())
