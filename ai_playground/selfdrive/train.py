import time

import click
import tensorflow as tf

from tf_agents.agents.tf_agent import TFAgent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from ai_playground.selfdrive.agents import PPOAgent, HAgent
from ai_playground.selfdrive.environments import SelfDriveEnvironment


class AlgorithmManager:
    def __init__(self, ctx: click.Context):
        self.ctx = ctx
        self.tf_env: tf_py_environment.TFPyEnvironment = tf_py_environment.TFPyEnvironment(SelfDriveEnvironment(ctx))
        self.train_step_counter = tf.compat.v1.train.get_or_create_global_step()
        self.tf_agent: TFAgent = self.get_agent(
            algorithm_type=ctx.obj['config']['algorithm']['name']
        )
        self.tf_agent.initialize()

    def get_agent(self, algorithm_type: str):
        if algorithm_type == 'ppo':
            agent = PPOAgent(self.ctx).create_ppo_agent(self.tf_env, self.train_step_counter)
            return agent
        if algorithm_type == 'hrl':
            pass
            # agent = HAgent(self.ctx)
            # return agent


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


