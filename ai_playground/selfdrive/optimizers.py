import tensorflow as tf


def get_optimizer(optimizer_type: str, params: dict):
    if optimizer_type == 'adam':
        return tf.compat.v1.train.AdamOptimizer(
            learning_rate=params['learning_rate'],
            epsilon=params['epsilon']
        )
