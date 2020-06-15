from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import tensorflow as tf
from tf_agents.networks.encoding_network import EncodingNetwork


class ICModule:
    def __init__(self, ctx: click.Context):
        self.ctx = ctx
        self.icm_params = self.ctx.obj['icm_params']
        self.icm_optim = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.icm_params['learning_rate'],
            epsilon=self.icm_params['epsilon']
        )
        self.loss_obj = tf.keras.losses.mean_squared_error()
        self.forward_model = None
        self.inverse_model = None
        self.intrinsic_reward = 0

    def train_ic_module(self, trajectory):
        train_loss_results = []
        train_accuracy_results = []

    def get_intrinsic_reward(self):
        return self.intrinsic_reward

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def loss(self, model, x, y, training):
        y_ = model(x, training=training)
        return self.loss_obj(y_true=y, y_pred=y_)

    def self_loss(self):
        pass
