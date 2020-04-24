import time

import click
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec

from ai_playground.selfdrive.environment import SelfDriveEnvironment


class AlgorithmManager:
    def __init__(self, ctx: click.Context):
        self.tf_env: tf_py_environment.TFPyEnvironment = tf_py_environment.TFPyEnvironment(SelfDriveEnvironment(ctx))
        self.train_step_counter = tf.compat.v1.train.get_or_create_global_step()
        self.tf_agent: TFAgent = self.get_agent(
            algo_type=ctx.obj['config']['algorithm']['algo_type'],
            params=ctx.obj['config']['algorithm']['params'][ctx.obj['config']['algorithm']['algo_type']]
        )
        self.tf_agent.initialize()

    def get_agent(self, algo_type: str, params: dict):
        if algo_type == 'ppo':
            agent = ppo_agent.PPOAgent(
                time_step_spec=self.tf_env.time_step_spec(),
                action_spec=self.tf_env.action_spec(),
                optimizer=self.get_optimizer(params['optimizer'], params['optimizer_params'][params['optimizer']]),
                actor_net=self.create_netowork(params['ppo_actor_net']),
                value_net=self.create_netowork(params['ppo_value_net']),
                num_epochs=params['num_epochs'],
                train_step_counter=self.train_step_counter,
                discount_factor=params['discount_factor'],
                gradient_clipping=params['gradient_clipping'],
                entropy_regularization=params['entropy_regularization'],
                importance_ratio_clipping=params['importance_ratio_clipping'],
                use_gae=params['use_gae'],
                use_td_lambda_return=params['use_td_lambda_return']
            )
            return agent
        if algo_type == 'hrl':
            return 1

    def create_netowork(self, network_type: str):
        if network_type == 'actor_rnn':
            return ActorDistributionRnnNetwork(
                input_tensor_spec=self.tf_env.observation_spec(),
                output_tensor_spec=self.tf_env.action_spec(),
                conv_layer_params=[(16, 8, 4), (32, 4, 2)],
                input_fc_layer_params=(256, ),
                lstm_size=(256, ),
                output_fc_layer_params=(128, ),
                activation_fn=tf.nn.elu
            )
        if network_type == 'value_rnn':
            return ValueRnnNetwork(
                input_tensor_spec=self.tf_env.observation_spec(),
                conv_layer_params=[(16, 8, 4), (32, 4, 2)],
                input_fc_layer_params=(256, ),
                lstm_size=(256, ),
                output_fc_layer_params=(128, ),
                activation_fn=tf.nn.elu
            )

    def get_optimizer(self, optimizer_type: str, params: dict):
        if optimizer_type == 'adam':
            return tf.compat.v1.train.AdamOptimizer(
                learning_rate=params['learning_rate'],
                epsilon=params['epsilon']
            )


class Trainer:
    def __init__(self, ctx: click.Context):
        self.algo_manager = AlgorithmManager(ctx)
        self.ctx = ctx

    def train(self):
        env_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            env_steps_metric
        ]

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.algo_manager.tf_agent.collect_data_spec,
            batch_size=1,
            max_length=self.ctx.obj['config']['train_session']['replay_buff_capacity']
        )
        collect_dirver = dynamic_episode_driver.DynamicEpisodeDriver(
            env=self.algo_manager.tf_env,
            policy=self.algo_manager.tf_agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_episodes=self.ctx.obj['config']['train_session']['collect_eps_per_iter']
        )
        collect_time = 0
        train_time = 0
        while env_steps_metric.result() < self.ctx.obj['config']['train_session']['num_env_steps']:
            start_time = time.time()
            collect_dirver.run()
            collect_time += time.time() - start_time

            start_time = time.time()
            trajectories = replay_buffer.gather_all()
            total_loss, _ = self.algo_manager.tf_agent.train(experience=trajectories)
            replay_buffer.clear()
            train_time += time.time() - start_time
            print(total_loss)


