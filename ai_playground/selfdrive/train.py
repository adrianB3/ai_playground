import os
import time
import neptune
import click
import tensorflow as tf
import sys

from tf_agents import environments
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer

from ai_playground.selfdrive.agents import PPOAgent, HAgent
from ai_playground.selfdrive.environments import SelfDriveEnvironment
from ai_playground.utils.logger import get_logger
from ai_playground.utils.exp_data import create_exp_local, init_neptune

logger = get_logger()


class AlgorithmManager:
    def __init__(self, ctx: click.Context):
        self.ctx = ctx
        self.tf_env: tf_py_environment.TFPyEnvironment = tf_py_environment.TFPyEnvironment(SelfDriveEnvironment(ctx))
        # environments.utils.validate_py_environment(self.tf_env)
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
        if ctx.obj['log2neptune']:
            self.neptune_exp = init_neptune(ctx)
            logger.info("This experiment is logged to neptune.ai")
        self.exp_dir = create_exp_local(
            experiments_dir=self.ctx.obj['config']['utils']['experiments_dir'],
            exp_name=self.ctx.obj['exp_name']
        )
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.algo_manager.tf_agent.collect_data_spec,
            batch_size=1,
            max_length=self.ctx.obj['config']['train_session']['replay_buff_capacity']
        )
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.checkpointer = Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.algo_manager.tf_agent,
            policy=self.algo_manager.tf_agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.algo_manager.train_step_counter
        )
        self.tf_policy_saver_dir = os.path.join(self.exp_dir, 'policy')
        self.tf_policy_saver = PolicySaver(self.algo_manager.tf_agent.policy)

    def save_artifacts(self):
        self.checkpointer.save(self.algo_manager.train_step_counter)
        self.tf_policy_saver.save(self.tf_policy_saver_dir)
        logger.info("Saved artifacts at: " + self.exp_dir)

    def exit_gracefully(self):
        logger.warning("Training interrupted. Saving policy...")
        # TODO - save policy
        if self.ctx.obj['log2neptune']:
            self.neptune_exp.stop()
        sys.exit(0)

    def train(self):
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()
        ]
        replay_observer = [self.replay_buffer.add_batch]
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=4,
            sample_batch_size=1,
            num_steps=2
        ).prefetch(3)
        iterator = iter(dataset)
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            env=self.algo_manager.tf_env,
            policy=self.algo_manager.tf_agent.collect_policy,
            observers=replay_observer + step_metrics,
            num_episodes=self.ctx.obj['config']['train_session']['collect_eps_per_iter']
        )
        self.algo_manager.tf_agent.train_step_counter.assign(0)
        collect_time = 0
        train_time = 0
        returns = []
        for _ in range(self.ctx.obj['config']['train_session']['num_env_steps']):
            for _ in range(self.ctx.obj['config']['train_session']['collect_eps_per_iter']):
                collect_driver.run()
            experience, unused_info = next(iterator)
            train_loss = self.algo_manager.tf_agent.train(experience)

            step = self.algo_manager.tf_agent.train_step_counter.numpy()
            self.save_artifacts()
            if self.ctx.obj['log2neptune']:
                pass

        # try:
        #     while env_steps_metric.result() < :
        #         start_time = time.time()
        #         collect_driver.run()
        #         collect_time += time.time() - start_time
        #
        #         start_time = time.time()
        #         trajectories = self.replay_buffer.gather_all()
        #         total_loss, _ = self.algo_manager.tf_agent.train(experience=trajectories)
        #         self.replay_buffer.clear()
        #         train_time += time.time() - start_time
        #
        #
        #             neptune.log_metric("collect_time", collect_time)
        # except (KeyboardInterrupt, SystemExit):
        #     self.exit_gracefully()
