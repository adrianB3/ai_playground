import os

import click
import tensorflow as tf
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from tf_agents.agents import PPOAgent
from tf_agents.drivers import driver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from ai_playground.selfdrive.environments import UnityEnv, HighLvlEnv, LowLvlEnv
from ai_playground.selfdrive.optimizers import get_optimizer
from ai_playground.utils.exp_data import create_exp_local
from ai_playground.utils.logger import get_logger

logger = get_logger()


class SelfDriveAgent:
    def __init__(self, ctx: click.Context):
        self.ctx = ctx
        self.exp_dir = create_exp_local(
            experiments_dir=self.ctx.obj['config']['utils']['experiments_dir'],
            exp_name=self.ctx.obj['exp_name']
        )
        unity_env: UnityEnvironment = UnityEnv(ctx).get_env()
        unity_env.reset()
        self.high_lvl_env: TFPyEnvironment = TFPyEnvironment(HighLvlEnv(self.ctx, unity_env))
        self.low_lvl_env: TFPyEnvironment = TFPyEnvironment(LowLvlEnv(self.ctx, unity_env))

        self.haar_config = self.ctx.obj['config']['algorithm']['params']['haar']
        self.optim_config = self.ctx.obj['config']['algorithm']['params'][
            self.ctx.obj['config']['algorithm']['params']['haar']['policy_optimizer']]

        self.high_lvl_agent: PPOAgent = PPOAgent(
            time_step_spec=self.high_lvl_env.time_step_spec(),
            action_spec=self.high_lvl_env.action_spec(),
            optimizer=get_optimizer(self.optim_config['optimizer'],
                                    self.optim_config['optimizer_params'][self.optim_config['optimizer']]),
            actor_net=create_network(self.optim_config['ppo_actor_net'], self.high_lvl_env.observation_spec(),
                                     self.high_lvl_env.action_spec()),
            value_net=create_network(self.optim_config['ppo_value_net'], self.high_lvl_env.observation_spec(),
                                     self.high_lvl_env.action_spec()),
            importance_ratio_clipping=self.optim_config['importance_ratio_clipping'],
            discount_factor=self.haar_config['discount_factor_high_lvl'],
            entropy_regularization=self.optim_config['entropy_regularization'],
            num_epochs=self.optim_config['num_epochs'],
            use_gae=self.optim_config['use_gae'],
            use_td_lambda_return=self.optim_config['use_td_lambda_return'],
            gradient_clipping=self.optim_config['gradient_clipping'],
            train_step_counter=tf.compat.v2.Variable(0)
        )

        self.low_lvl_agent: PPOAgent = PPOAgent(
            time_step_spec=self.low_lvl_env.time_step_spec(),
            action_spec=self.low_lvl_env.action_spec(),
            optimizer=get_optimizer(self.optim_config['optimizer'],
                                    self.optim_config['optimizer_params'][self.optim_config['optimizer']]),
            actor_net=create_network(self.optim_config['ppo_actor_net'], self.low_lvl_env.observation_spec(),
                                     self.low_lvl_env.action_spec()),
            value_net=create_network(self.optim_config['ppo_value_net'], self.low_lvl_env.observation_spec(),
                                     self.low_lvl_env.action_spec()),
            importance_ratio_clipping=self.optim_config['importance_ratio_clipping'],
            discount_factor=self.haar_config['discount_factor_low_lvl'],
            entropy_regularization=self.optim_config['entropy_regularization'],
            num_epochs=self.optim_config['num_epochs'],
            use_gae=self.optim_config['use_gae'],
            use_td_lambda_return=self.optim_config['use_td_lambda_return'],
            gradient_clipping=self.optim_config['gradient_clipping'],
            train_step_counter=tf.compat.v2.Variable(0)
        )
        self.high_lvl_agent.initialize()
        self.low_lvl_agent.initialize()

    def train(self):
        # tf.keras.utils.plot_model(
        #     self.high_lvl_agent.actor_net, to_file='model.png', show_shapes=True, show_layer_names=True,
        #     rankdir='TB', expand_nested=True, dpi=300
        # )
        k_0 = self.haar_config['k_0']
        k_s = self.haar_config['k_s']
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()
        ]
        for ep_count in range(0, self.haar_config['num_eps']):
            high_lvl_ts = self.high_lvl_env.current_time_step()
            # high_lvl_ps = self.high_lvl_agent.collect_policy.get_initial_state(self.high_lvl_env.batch_size)
            print()


def create_network(name: str, input_spec, output_spec):
    preprocessing_layers = {
        'image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 4),
                                             tf.keras.layers.Flatten()]),
        'vector': tf.keras.layers.Dense(5)
    }
    preprocessing_combiner = tf.keras.models.Sequential(
        [tf.keras.layers.Concatenate(axis=-1), tf.keras.layers.Reshape((100, 250, 2))])

    if name == "actor_preproc":
        return ActorDistributionNetwork(
            input_tensor_spec=input_spec,
            output_tensor_spec=output_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            fc_layer_params=(128, 75),
            activation_fn=tf.nn.elu
        )

    if name == "value_preproc":
        return ValueNetwork(
            input_tensor_spec=input_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            fc_layer_params=(75, 40),
            activation_fn=tf.nn.elu
        )
