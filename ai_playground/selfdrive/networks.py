import tensorflow as tf
import numpy as np
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.networks import network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.categorical_projection_network import CategoricalProjectionNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork


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


def create_network(network_type: str, observation_spec, action_spec):
    preprocessing_layers = {
        'image': tf.keras.models.Sequential([
                                            tf.keras.layers.Conv2D(8, 4, input_shape=(100, 250, 1)),
                                            tf.keras.layers.Flatten()]),
        'vector': tf.keras.layers.Dense(4)
    }
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    if network_type == 'actor_rnn':
        return ActorDistributionRnnNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            input_fc_layer_params=(256, ),
            lstm_size=(256, ),
            output_fc_layer_params=(128, ),
            activation_fn=tf.nn.elu
        )
    if network_type == 'value_rnn':
        return ValueRnnNetwork(
            input_tensor_spec=observation_spec,
            conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            input_fc_layer_params=(256, ),
            lstm_size=(256, ),
            output_fc_layer_params=(128, ),
            activation_fn=tf.nn.elu
        )
    if network_type == 'actor_rnn_pre':
        return ActorDistributionRnnNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            input_fc_layer_params=(256, ),
            lstm_size=(256, ),
            output_fc_layer_params=(128, ),
            activation_fn=tf.nn.elu,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
        )
    if network_type == 'value_rnn_pre':
        return ValueRnnNetwork(
            input_tensor_spec=observation_spec,
            #conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            input_fc_layer_params=(256, ),
            lstm_size=(256, ),
            output_fc_layer_params=(128, ),
            activation_fn=tf.nn.elu,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner
        )

    if network_type == 'actor_pre_proj':
        model = ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            #conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            fc_layer_params=(128,),
            activation_fn=tf.nn.elu,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            # discrete_projection_net=CategoricalProjectionNetwork(
            #     sample_spec=action_spec
            # )
        )
        # tf.keras.utils.plot_model(
        #     model, to_file='model.png', show_shapes=True, show_layer_names=True,
        #     rankdir='TB', expand_nested=True, dpi=300
        # )
        return model
