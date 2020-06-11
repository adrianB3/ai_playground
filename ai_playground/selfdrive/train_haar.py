import os

import click
import gin
import tensorflow as tf
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from tf_agents.agents import PPOAgent
from tf_agents.drivers import driver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver, is_bandit_env
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import time_step as ts, trajectory
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import nest_utils, common
from tf_agents.utils.common import Checkpointer
from tf_agents.eval.metric_utils import log_metrics, MetricsGroup

from ai_playground.selfdrive.haar_ppo_agent import HaarPPOAgent
from ai_playground.selfdrive.environments import UnityEnv, HighLvlEnv, LowLvlEnv
from ai_playground.selfdrive.optimizers import get_optimizer
from ai_playground.selfdrive.train_drivers import HaarLowLvlDriver
from ai_playground.utils.exp_data import create_exp_local, init_neptune
from ai_playground.utils.logger import get_logger

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')
logger = get_logger()


class SelfDriveAgent:
    def __init__(self, ctx: click.Context):
        self.ctx = ctx
        if self.ctx.obj['config']['utils']['load_exp']:
            self.exp_dir = self.ctx.obj['config']['utils']['load_exp']
            logger.info("Loading experiment from: " + self.exp_dir)
        else:
            self.exp_dir = create_exp_local(
                experiments_dir=self.ctx.obj['config']['utils']['experiments_dir'],
                exp_name=self.ctx.obj['exp_name']
            )

        if ctx.obj['log2neptune']:
            self.neptune_exp = init_neptune(ctx)
            logger.info("This experiment is logged to neptune.ai")

        self.unity_env: UnityEnvironment = UnityEnv(ctx).get_env()
        self.unity_env.reset()
        self.high_lvl_env: TFPyEnvironment = TFPyEnvironment(HighLvlEnv(self.ctx, self.unity_env))
        self.low_lvl_env: TFPyEnvironment = TFPyEnvironment(LowLvlEnv(self.ctx, self.unity_env))

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
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0)
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
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0)
        )

        self.high_lvl_agent.train_step_counter.assign(0)
        self.low_lvl_agent.train_step_counter.assign(0)
        self.high_lvl_agent.initialize()
        self.low_lvl_agent.initialize()

        self.low_lvl_replay_buffer = TFUniformReplayBuffer(
            data_spec=self.low_lvl_agent.collect_data_spec,
            batch_size=1,
            max_length=1000
        )
        self.high_lvl_replay_buffer = TFUniformReplayBuffer(
            data_spec=self.high_lvl_agent.collect_data_spec,
            batch_size=1,
            max_length=1000
        )
        self.modified_low_lvl_rep_buffer = TFUniformReplayBuffer(
            data_spec=self.low_lvl_agent.collect_data_spec,
            batch_size=1,
            max_length=1000
        )

        train_dir = os.path.join(self.exp_dir, 'train')
        eval_dir = os.path.join(self.exp_dir, 'eval')

        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=self.ctx.obj['config']['train_session']['summaries_flush_secs'] * 1000)
        # self.train_summary_writer.set_as_default()
        #
        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=self.ctx.obj['config']['train_session']['summaries_flush_secs'] * 1000)

        self.h_eval_metrics = [
            tf_metrics.ChosenActionHistogram()
        ]

        self.l_eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=self.ctx.obj['config']['train_session']['num_eval_episodes']),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.ctx.obj['config']['train_session']['num_eval_episodes'])
        ]

        self.h_step_observers = [
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(batch_size=1)
        ]

        self.l_step_observers = [
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(batch_size=1)
        ]

        self.h_checkpoint_dir = os.path.join(self.exp_dir, 'h_checkpoints')
        self.h_checkpointer = Checkpointer(
            ckpt_dir=self.h_checkpoint_dir,
            max_to_keep=1,
            agent=self.high_lvl_agent,
            policy=self.high_lvl_agent.policy,
            replay_buffer=self.high_lvl_replay_buffer,
            global_step=self.high_lvl_agent.train_step_counter,
            metrics=MetricsGroup(self.h_step_observers, 'high_lvl_train_metrics')
        )
        self.h_tf_policy_saver_dir = os.path.join(self.exp_dir, 'h_policy')
        self.h_tf_policy_saver = PolicySaver(self.high_lvl_agent.policy)

        self.l_checkpoint_dir = os.path.join(self.exp_dir, 'l_checkpoints')
        self.l_checkpointer = Checkpointer(
            ckpt_dir=self.l_checkpoint_dir,
            max_to_keep=1,
            agent=self.low_lvl_agent,
            policy=self.low_lvl_agent.policy,
            replay_buffer=self.modified_low_lvl_rep_buffer,
            global_step=self.low_lvl_agent.train_step_counter,
            metrics=MetricsGroup(self.l_step_observers, 'low_lvl_train_metrics')
        )
        self.l_tf_policy_saver_dir = os.path.join(self.exp_dir, 'l_policy')
        self.l_tf_policy_saver = PolicySaver(self.low_lvl_agent.policy)

        self.h_checkpointer.initialize_or_restore()
        self.l_checkpointer.initialize_or_restore()

        self.k_0 = self.haar_config['k_0']
        self.k_s = self.haar_config['k_s']
        self.j = self.haar_config['j']

    def train(self):

        high_lvl_driver = DynamicStepDriver(
            env=self.high_lvl_env,
            policy=self.high_lvl_agent.collect_policy,
            observers=[self.high_lvl_replay_buffer.add_batch] + self.h_step_observers,
            num_steps=1
        )
        low_lvl_driver = HaarLowLvlDriver(
            env=self.low_lvl_env,
            policy=self.low_lvl_agent.collect_policy,
            observers=[self.low_lvl_replay_buffer.add_batch] + self.l_step_observers,
            num_steps=1
        )

        high_lvl_driver._num_steps = 1
        high_lvl_driver.run()
        high_lvl_dataset = self.high_lvl_replay_buffer.as_dataset(sample_batch_size=1)
        high_lvl_iter = iter(high_lvl_dataset)
        h_exp_prev, _ = next(high_lvl_iter)

        for ep_count in range(1, self.haar_config['num_eps'] + 1):
            sds = list(self.unity_env.side_channels.values())
            sds[1].set_float_parameter("num_steps", self.k_0 * self.j)
            high_lvl_action = h_exp_prev.action
            low_lvl_driver._num_steps = self.k_0
            low_lvl_driver.run(high_lvl_action=high_lvl_action)

            if ep_count % self.j == 0:
                exp = self.high_lvl_replay_buffer.gather_all()
                self.high_lvl_agent.train(exp)

                self.h_checkpointer.save(self.high_lvl_agent.train_step_counter)
                self.h_tf_policy_saver.save(self.h_tf_policy_saver_dir)

                self.l_checkpointer.save(self.low_lvl_agent.train_step_counter)
                self.l_tf_policy_saver.save(self.l_tf_policy_saver_dir)

                logger.info("Saved artifacts at: " + self.exp_dir)

                self.high_lvl_env.reset()
                self.unity_env.reset()
                self.high_lvl_replay_buffer.clear()
                print("High level trained.")

            # modified_rewards = tf.math.scalar_mul(1 / self.k_0, advs)
            # sliced_mod_rewards = []
            # for i in range(0, int(self.k_0) - 1):
            #     slice = tf.slice(modified_rewards, [0, i], [-1, 1])
            #     sliced_mod_rewards.append(slice)

            # new_low_lvl_dataset: MapDataset = MapDataset()
            # for el in low_lvl_dataset:
            #     new_trajectory = Trajectory(el[0].step_type, el[0].observation, el[0].action, el[0].policy_info,
            #                                 el[0].next_step_type, tf.math.scalar_mul(1/k_0, el[0].reward), el[0].discount)
            #     new_buff_info = el[1]
            #     new_dataset_el = (new_trajectory, new_buff_info)
            #     print(new_dataset_el)
            # low_lvl_dataset = low_lvl_dataset.map(lambda traj_info: (tf.math.scalar_mul(1/k_0, traj_info[0].reward), traj_info[1]))

            # for el in low_lvl_dataset: #todo, not same nest structure
            #     values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), el)
            #     new_low_lvl_rep_buffer.add_batch(values_batched)

            # mod_reward = None
            # for i in range(0, self.haar_config['num_low_lvl_steps']):
            #     l_exp, _ = next(low_lvl_iter)
            #     if i % k_0 == 0:
            #         if i / k_0 < len(sliced_mod_rewards):
            #             mod_reward = sliced_mod_rewards[int(i / k_0)]
            #         else:
            #             mod_reward = l_exp.reward
            #     rew = mod_reward.numpy()
            #     rew = np.asscalar(rew)
            #
            #     indices = tf.constant([[1]])
            #     updates = tf.constant([rew])
            #     tf.tensor_scatter_nd_update(l_exp.reward, indices, updates)
            low_lvl_dataset = self.low_lvl_replay_buffer.as_dataset(sample_batch_size=1)
            iterator = iter(low_lvl_dataset)

            # calculare r_t^h
            low_lvl_cumulative_reward = 0
            for _ in range(self.k_0):
                traj, _ = next(iterator)
                low_lvl_cumulative_reward += traj.reward.numpy()

            high_lvl_driver._num_steps = 1
            high_lvl_driver.run()
            high_lvl_dataset = self.high_lvl_replay_buffer.as_dataset(sample_batch_size=1)
            high_lvl_iter = iter(high_lvl_dataset)
            h_exp_current, _ = next(high_lvl_iter)

            self.advantage = self.calculate_advantage(low_lvl_cumulative_reward, h_exp_current, h_exp_prev)

            low_lvl_dataset = low_lvl_dataset.map(self.modify_reward)
            iterator = iter(low_lvl_dataset)
            for _ in range(self.k_0):
                transition, _ = next(iterator)
                values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), transition)
                self.modified_low_lvl_rep_buffer.add_batch(values_batched)

            low_lvl_exp = self.modified_low_lvl_rep_buffer.gather_all()
            loss = self.low_lvl_agent.train(low_lvl_exp)

            if ep_count % 5 == 0:
                avg_return = self.haar_compute_avg_reward(5)
                print("Average return: " + str(avg_return))

            print("Low lvl Loss: " + str(loss.loss.numpy()))
            # print(self.l_step_observers[1].result().numpy())
            # print(self.h_step_observers[1].result().numpy())

            if self.ctx.obj['log2neptune']:
                pass

            h_exp_prev = h_exp_current
            self.low_lvl_env.reset()
            self.low_lvl_replay_buffer.clear()
            self.modified_low_lvl_rep_buffer.clear()

    def modify_reward(self, traj, buff):
        new_trajectory = Trajectory(traj.step_type, traj.observation, traj.action, traj.policy_info,
                                    traj.next_step_type, self.advantage, traj.discount)
        return (new_trajectory, buff)

    def calculate_advantage(self, cumulative_reward, h_experience_current, h_experience_prev):
        h_value_estimate_current, _ = self.high_lvl_agent.collect_policy.apply_value_network(
            h_experience_current.observation, h_experience_current.step_type, value_state=(),
            training=False)
        h_value_estimate_prev, _ = self.high_lvl_agent.collect_policy.apply_value_network(
            h_experience_prev.observation, h_experience_prev.step_type, value_state=(),
            training=False)
        adv = (1/self.k_0) * (cumulative_reward + self.haar_config['discount_factor_high_lvl'] * h_value_estimate_current.numpy() - h_value_estimate_prev.numpy())
        return adv

    def haar_compute_avg_reward(self, num_eps):
        total_return = 0.0
        for _ in range(num_eps):
            time_step_h = self.high_lvl_env.reset()
            time_step_l = self.low_lvl_env.reset()
            self.unity_env.reset()
            ep_return = 0.0

            h_action = self.high_lvl_agent.policy.action(time_step_h)
            for i in range(self.k_0):
                low_lvl_obs = {'image': time_step_l.observation['image'], 'vector': h_action.action}
                time_step_l = time_step_l._replace(
                    observation=low_lvl_obs
                )
                l_action = self.low_lvl_agent.policy.action(time_step_l)
                time_step_l = self.low_lvl_env.step(l_action)
                ep_return += time_step_l.reward
            self.low_lvl_env.reset()
            self.high_lvl_env.reset()
            self.unity_env.reset()
            total_return += ep_return

        avg_return = total_return / num_eps
        return avg_return.numpy()[0]


def create_network(name: str, input_spec, output_spec):
    preprocessing_layers = {
        'image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(4, (3, 3)),
                                             tf.keras.layers.Conv2D(8, (3, 3)),
                                             tf.keras.layers.Flatten()]),
        'vector': tf.keras.layers.Dense(4)
    }
    preprocessing_combiner = tf.keras.models.Sequential(
        [tf.keras.layers.Concatenate(axis=-1)])

    if name == "actor_preproc":
        return ActorDistributionNetwork(
            input_tensor_spec=input_spec,
            output_tensor_spec=output_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            # conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            fc_layer_params=(128, 75),
            activation_fn=tf.nn.elu
        )

    if name == "value_preproc":
        return ValueNetwork(
            input_tensor_spec=input_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            # conv_layer_params=[(16, 8, 4), (32, 4, 2)],
            fc_layer_params=(75, 40),
            activation_fn=tf.nn.elu
        )
