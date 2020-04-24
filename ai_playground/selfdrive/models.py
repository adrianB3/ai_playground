import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='dqn')
        self.fc1 = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.val_adv_fun = tf.keras.layers.Dense(num_actions + 1, activation='relu', kernel_initializer='he_uniform')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        val_adv = self.val_adv_fun(x)

        outputs = tf.expand_dims(val_adv[:, 0], -1) + (val_adv[:, 1:] - tf.reduce_mean(val_adv[:, 1:], -1, keepdims=True))
        return outputs

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]

