from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from ast import literal_eval
from typing import Tuple

import click
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent, ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment, wrappers
from tf_agents.environments import tf_py_environment
from tf_agents.environments.wrappers import RunStats
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from ai_playground.selfdrive.environments import SelfDriveEnvironment
from ai_playground.utils.exp_data import create_exp_local, init_neptune
from ai_playground.utils.logger import get_logger

logger = get_logger()


class PPOTrainer:
    def __init__(self, ctx: click.Context):
        self.ctx = ctx
        self.ppo_config = self.ctx.obj['config']['algorithm']['params']['ppo']
        self.optimizer_name = self.ctx.obj['config']['algorithm']['params']['ppo']['optimizer']
        self.train_sess_config = self.ctx.obj['config']['train_session']

        self.exp_dir, self.neptune_exp = self.prep_experiment()
        self.train_dir = os.path.join(self.exp_dir, 'train')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')
        self.saved_model_dir = os.path.join(self.exp_dir, 'policy_saved_model')
        self.train_summary_writer, self.eval_summary_writer = self.create_summary_writers()

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        if self.train_sess_config['run_parallel']:
            # self.env = parallel_py_environment.ParallelPyEnvironment(
            #     [lambda: env_load_fn(env_name)] * self.train_sess_config['']))
            pass
        else:
            pyenv = SelfDriveEnvironment(ctx)
            pyenv_limit = wrappers.TimeLimit(pyenv, duration=3000)
            pyenv_stats = wrappers.RunStats(pyenv_limit)
            self.env = tf_py_environment.TFPyEnvironment(pyenv_stats)
        # self.eval_env = tf_py_environment.TFPyEnvironment(SelfDriveEnvironment(ctx))
        self.agent = self.create_ppo_agent()
        self.environment_steps_metric = tf_metrics.EnvironmentSteps()
        self.step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            self.environment_steps_metric,
        ]

        self.train_metrics = self.step_metrics + [
            tf_metrics.AverageReturnMetric(
                batch_size=1, buffer_size=self.train_sess_config['collect_eps_per_iter']),  # todo adapt to parallel stuff
            tf_metrics.AverageEpisodeLengthMetric(
                batch_size=1, buffer_size=self.train_sess_config['collect_eps_per_iter'])
        ]

        self.eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=self.train_sess_config['num_eval_episodes']),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.train_sess_config['num_eval_episodes'])
        ]

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=1,  # todo adapt to parallel stuff
            max_length=self.train_sess_config['replay_buff_capacity'])

        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.train_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
            metrics=metric_utils.MetricsGroup(self.train_metrics, 'train_metrics'))
        self.policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(self.eval_dir, 'policy'),
            policy=self.eval_policy,
            global_step=self.global_step)
        self.saved_model = policy_saver.PolicySaver(
            self.eval_policy, train_step=self.global_step)

        self.train_checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_episodes=self.train_sess_config['collect_eps_per_iter'])

        if self.train_sess_config['use_tf_function']:
            self.collect_driver.run = common.function(self.collect_driver.run, autograph=False)
            self.agent.train = common.function(self.agent.train, autograph=False)
            self.train_step = common.function(self.train_step)

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    def train_step(self):
        trajectories = self.replay_buffer.gather_all()
        return self.agent.train(experience=trajectories)

    def prep_experiment(self):
        if self.ctx.obj['config']['utils']['load_exp']:
            exp_dir = self.ctx.obj['config']['utils']['load_exp']
            logger.info("Loading experiment from: " + exp_dir)
        else:
            exp_dir = create_exp_local(
                experiments_dir=self.ctx.obj['config']['utils']['experiments_dir'],
                exp_name=self.ctx.obj['exp_name']
            )
        neptune_exp = None
        if self.ctx.obj['log2neptune']:
            neptune_exp = init_neptune(self.ctx)
            logger.info("This experiment is logged to neptune.ai")
        return exp_dir, neptune_exp

    def create_summary_writers(self):
        summaries_flush_secs = self.ctx.obj['config']['train_session']['summaries_flush_secs']
        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            self.train_dir, flush_millis=summaries_flush_secs * 1000)
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            self.eval_dir, flush_millis=summaries_flush_secs * 1000)

        return train_summary_writer, eval_summary_writer

    def create_ppo_agent(self):
        tf_agent = ppo_clip_agent.PPOClipAgent(
            time_step_spec=self.env.time_step_spec(),
            action_spec=self.env.action_spec(),
            optimizer=self.get_optimizer(self.optimizer_name),

            actor_net=self.get_networks(self.ppo_config['ppo_actor_net']),
            value_net=self.get_networks(self.ppo_config['ppo_value_net']),
            importance_ratio_clipping=self.ppo_config['importance_ratio_clipping'],
            discount_factor=self.ppo_config['discount_factor'],
            entropy_regularization=self.ppo_config['entropy_regularization'],
            normalize_observations=self.ppo_config['normalize_observations'],
            normalize_rewards=self.ppo_config['normalize_rewards'],
            use_gae=self.ppo_config['use_gae'],
            use_td_lambda_return=self.ppo_config['use_td_lambda_return'],
            num_epochs=self.ppo_config['num_epochs'],
            gradient_clipping=self.ppo_config['gradient_clipping'],
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=self.global_step
        )
        tf_agent.initialize()
        return tf_agent

    def get_networks(self, name: str):
        networks_params = self.ctx.obj['config']['algorithm']['networks_params']

        if not self.ctx.obj['config']['environment']['only_visual_obs']:
            preprocessing_layers = {
                'image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
                                                     tf.keras.layers.MaxPooling2D((2, 2)),
                                                     tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                                                     tf.keras.layers.Flatten()]),
                'vector1': tf.keras.layers.Dense(12, activation='relu')
            }
            preprocessing_combiner = tf.keras.models.Sequential(
                [tf.keras.layers.Concatenate(axis=-1)])

            if name == "actor_preproc":
                return ActorDistributionNetwork(
                    input_tensor_spec=self.env.observation_spec(),
                    output_tensor_spec=self.env.action_spec(),
                    preprocessing_layers=preprocessing_layers,
                    preprocessing_combiner=preprocessing_combiner,
                    fc_layer_params=literal_eval(networks_params['actor_fc_layers']),
                    activation_fn=tf.nn.elu
                )

            if name == "value_preproc":
                return ValueNetwork(
                    input_tensor_spec=self.env.observation_spec(),
                    preprocessing_layers=preprocessing_layers,
                    preprocessing_combiner=preprocessing_combiner,
                    fc_layer_params=literal_eval(networks_params['value_fc_layers']),
                    activation_fn=tf.nn.elu)
        else:
            if name == "actor_preproc":
                return ActorDistributionNetwork(
                    input_tensor_spec=self.env.observation_spec(),
                    output_tensor_spec=self.env.action_spec(),
                    conv_layer_params=[(16, 8, 2), (32, 4, 2), (32, 4, 2)],
                    fc_layer_params=literal_eval(networks_params['actor_fc_layers']),
                    activation_fn=tf.nn.elu
                )

            if name == "value_preproc":
                return ValueNetwork(
                    input_tensor_spec=self.env.observation_spec(),
                    conv_layer_params=[(16, 8, 2), (16, 4, 2), (32, 4, 2)],
                    fc_layer_params=literal_eval(networks_params['value_fc_layers']),
                    activation_fn=tf.nn.elu
                )

    def get_optimizer(self, name: str):
        optim_config = self.ctx.obj['config']['algorithm']['params']['ppo']['optimizer_params'][name]
        if name == 'adam':
            return tf.compat.v1.train.AdamOptimizer(
                learning_rate=optim_config['learning_rate'],
                epsilon=optim_config['epsilon']
            )

    def train(self):
        with tf.compat.v2.summary.record_if(
                lambda: tf.math.equal(self.global_step % self.train_sess_config['summary_interval'], 0)):
            collect_time = 0
            train_time = 0
            timed_at_step = self.global_step.numpy()

            while self.environment_steps_metric.result() < self.train_sess_config['num_env_steps']:
                global_step_val = self.global_step.numpy()
                if global_step_val % self.train_sess_config['eval_interval'] == 0:
                    metric_utils.eager_compute(
                        self.eval_metrics,
                        self.env,
                        self.eval_policy,
                        num_episodes=self.train_sess_config['num_eval_episodes'],
                        train_step=self.global_step,
                        summary_writer=self.eval_summary_writer,
                        summary_prefix='Metrics'
                    )
                    print("Evaluated.")
                for train_metric in self.train_metrics:
                    train_metric.tf_summaries(
                        train_step=self.global_step, step_metrics=self.step_metrics)

                start_time = time.time()
                self.collect_driver.run()
                collect_time += time.time() - start_time
                start_time = time.time()
                total_loss, _ = self.train_step()
                self.replay_buffer.clear()
                train_time += time.time() - start_time

                if global_step_val % self.train_sess_config['log_interval'] == 0:
                    logger.info('step = %d, loss = %f', global_step_val, total_loss)
                    steps_per_sec = (
                            (global_step_val - timed_at_step) / (collect_time + train_time))
                    logger.info('%.3f steps/sec', steps_per_sec)
                    logger.info('collect_time = %.3f, train_time = %.3f', collect_time, train_time)
                    logger.info('Env steps: ' + str(self.environment_steps_metric.result().numpy()))
                    with tf.compat.v2.summary.record_if(True):
                        tf.compat.v2.summary.scalar(
                            name='global_steps_per_sec', data=steps_per_sec, step=self.global_step)

                    if global_step_val % self.train_sess_config['train_checkpoint_interval'] == 0:
                        self.train_checkpointer.save(global_step=global_step_val)

                    if global_step_val % self.train_sess_config['policy_checkpoint_interval'] == 0:
                        self.policy_checkpointer.save(global_step=global_step_val)
                        saved_model_path = os.path.join(
                            self.saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
                        self.saved_model.save(saved_model_path)

                    if self.ctx.obj['log2neptune']:
                        self.neptune_exp.log_metric('step', global_step_val)
                        self.neptune_exp.log_metric('loss', global_step_val)
                        self.neptune_exp.log_metric('steps/sec', steps_per_sec)
                        self.neptune_exp.log_metric('collect_time', collect_time)
                        self.neptune_exp.log_metric('train_time', train_time)

                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

            # One final eval before exiting.
            metric_utils.eager_compute(
                self.eval_metrics,
                self.env,
                self.eval_policy,
                num_episodes=self.train_sess_config['num_eval_episodes'],
                train_step=self.global_step,
                summary_writer=self.eval_summary_writer,
                summary_prefix='Metrics'
            )
            if self.ctx.obj['log2neptune']:
                self.neptune_exp.close()

    def inference(self):
        metric_utils.eager_compute(
            self.eval_metrics,
            self.env,
            self.eval_policy,
            num_episodes=1,
            train_step=self.global_step,
            summary_writer=self.eval_summary_writer,
            summary_prefix='Metrics'
        )
