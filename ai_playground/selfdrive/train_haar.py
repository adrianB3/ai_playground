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
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import time_step as ts, trajectory
from tf_agents.utils import nest_utils, common

from ai_playground.selfdrive.haar_ppo_agent import HaarPPOAgent
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
        self.unity_env: UnityEnvironment = UnityEnv(ctx).get_env()
        self.unity_env.reset()
        self.high_lvl_env: TFPyEnvironment = TFPyEnvironment(HighLvlEnv(self.ctx, self.unity_env))
        self.low_lvl_env: TFPyEnvironment = TFPyEnvironment(LowLvlEnv(self.ctx, self.unity_env))

        self.haar_config = self.ctx.obj['config']['algorithm']['params']['haar']
        self.optim_config = self.ctx.obj['config']['algorithm']['params'][
            self.ctx.obj['config']['algorithm']['params']['haar']['policy_optimizer']]

        self.high_lvl_agent: HaarPPOAgent = HaarPPOAgent(
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
            train_step_counter=tf.Variable(0)
        )

        self.low_lvl_agent: HaarPPOAgent = HaarPPOAgent(
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
            train_step_counter=tf.Variable(0)
        )
        self.high_lvl_agent.train_step_counter.assign(0)
        self.low_lvl_agent.train_step_counter.assign(0)
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
        low_lvl_replay_buffer = TFUniformReplayBuffer(
            data_spec=self.low_lvl_agent.collect_data_spec,
            batch_size=1,
            max_length=1000
        )
        high_lvl_replay_buffer = TFUniformReplayBuffer(
            data_spec=self.high_lvl_agent.collect_data_spec,
            batch_size=1,
            max_length=1000
        )
        high_lvl_driver = DynamicStepDriver(
            env=self.high_lvl_env,
            policy=self.high_lvl_agent.collect_policy,
            observers=[high_lvl_replay_buffer.add_batch],
            num_steps=1
        )
        low_lvl_driver = HaarLowLvlDriver(
            env=self.low_lvl_env,
            policy=self.low_lvl_agent.collect_policy,
            observers=[low_lvl_replay_buffer.add_batch],
            num_steps=1
        )
        for ep_count in range(0, self.haar_config['num_eps']):
            for low_lvl_step_count in range(0, self.haar_config['num_low_lvl_steps']):
                if low_lvl_step_count % k_0 == 0:
                    high_lvl_driver._num_steps = 1
                    high_lvl_driver.run()
                    high_lvl_dataset = high_lvl_replay_buffer.as_dataset(sample_batch_size=1)
                    high_lvl_iter = iter(high_lvl_dataset)
                    h_exp, _ = next(high_lvl_iter)

                    high_lvl_action = h_exp.action
                    low_lvl_driver._num_steps = k_0
                    low_lvl_driver.run(high_lvl_action=high_lvl_action)

            exp = high_lvl_replay_buffer.gather_all()
            self.high_lvl_agent.train(exp)

            advs = self.high_lvl_agent.normalized_adv
            modified_rewards = tf.math.scalar_mul(1 / k_0, advs)
            sliced_mod_rewards = []
            for i in range(0, int(self.haar_config['num_low_lvl_steps'] / k_0) - 1):
                slice = tf.slice(modified_rewards, [0, i], [-1, 1])
                sliced_mod_rewards.append(slice)

            low_lvl_dataset = low_lvl_replay_buffer.as_dataset(sample_batch_size=1)
            low_lvl_iter = iter(low_lvl_dataset)
            mod_reward = None
            for i in range(0, self.haar_config['num_low_lvl_steps']):
                l_exp, _ = next(low_lvl_iter)
                if i % k_0 == 0:
                    if i / k_0 < len(sliced_mod_rewards):
                        mod_reward = sliced_mod_rewards[int(i / k_0)]
                    else:
                        mod_reward = l_exp.reward
                rew = mod_reward.numpy()
                rew = np.asscalar(rew)

                indices = tf.constant([[1]])
                updates = tf.constant([rew])
                tf.tensor_scatter_nd_update(l_exp.reward, indices, updates)

            low_lvl_exp = low_lvl_replay_buffer.gather_all()
            self.low_lvl_agent.train(low_lvl_exp)

            high_lvl_replay_buffer.clear()
            low_lvl_replay_buffer.clear()



@gin.configurable
class HaarLowLvlDriver(driver.Driver):
    """A driver that takes N steps in an environment using a tf.while_loop.

      The while loop will run num_steps in the environment, only counting steps that
      result in an environment transition, i.e. (time_step, action, next_time_step).
      If a step results in environment resetting, i.e. time_step.is_last() and
      next_time_step.is_first() (traj.is_boundary()), this is not counted toward the
      num_steps.

      As environments run batched time_steps, the counters for all batch elements
      are summed, and execution stops when the total exceeds num_steps. When
      batch_size > 1, there is no guarantee that exactly num_steps are taken -- it
      may be more but never less.

      This termination condition can be overridden in subclasses by implementing the
      self._loop_condition_fn() method.
      """

    def __init__(
            self,
            env,
            policy,
            observers=None,
            transition_observers=None,
            num_steps=1,
    ):
        """Creates a DynamicStepDriver.

        Args:
          env: A tf_environment.Base environment.
          policy: A tf_policy.Base policy.
          observers: A list of observers that are updated after every step in the
            environment. Each observer is a callable(time_step.Trajectory).
          transition_observers: A list of observers that are updated after every
            step in the environment. Each observer is a callable((TimeStep,
            PolicyStep, NextTimeStep)).
          num_steps: The number of steps to take in the environment.

        Raises:
          ValueError:
            If env is not a tf_environment.Base or policy is not an instance of
            tf_policy.Base.
        """
        super(HaarLowLvlDriver, self).__init__(env, policy, observers,
                                               transition_observers)
        self._num_steps = num_steps
        self._run_fn = common.function_in_tf1()(self._run)

    def _loop_condition_fn(self):
        """Returns a function with the condition needed for tf.while_loop."""

        def loop_cond(counter, *_):
            """Determines when to stop the loop, based on step counter.

            Args:
              counter: Step counters per batch index. Shape [batch_size] when
                batch_size > 1, else shape [].

            Returns:
              tf.bool tensor, shape (), indicating whether while loop should continue.
            """
            return tf.less(tf.reduce_sum(input_tensor=counter), self._num_steps)

        return loop_cond

    def _loop_body_fn(self):
        """Returns a function with the driver's loop body ops."""

        def loop_body(counter, time_step, policy_state, high_lvl_action):
            """Runs a step in environment.

            While loop will call multiple times.

            Args:
              counter: Step counters per batch index. Shape [batch_size].
              time_step: TimeStep tuple with elements shape [batch_size, ...].
              policy_state: Policy state tensor shape [batch_size, policy_state_dim].
                Pass empty tuple for non-recurrent policies.

            Returns:
              loop_vars for next iteration of tf.while_loop.
            """
            low_lvl_obs = {'image': time_step.observation['image'], 'vector': high_lvl_action}
            time_step = time_step._replace(
                observation=low_lvl_obs
            )
            action_step = self.policy.action(time_step, policy_state)
            policy_state = action_step.state
            next_time_step = self.env.step(action_step.action)
            next_low_lvl_obs = {'image': next_time_step.observation['image'], 'vector': high_lvl_action}
            next_time_step = next_time_step._replace(
                observation=next_low_lvl_obs
            )

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            observer_ops = [observer(traj) for observer in self._observers]
            transition_observer_ops = [
                observer((time_step, action_step, next_time_step))
                for observer in self._transition_observers
            ]
            with tf.control_dependencies(
                    [tf.group(observer_ops + transition_observer_ops)]):
                time_step, next_time_step, policy_state = tf.nest.map_structure(
                    tf.identity, (time_step, next_time_step, policy_state))

            # While loop counter should not be incremented for episode reset steps.
            counter += tf.cast(~traj.is_boundary(), dtype=tf.int32)

            return [counter, next_time_step, policy_state, high_lvl_action]

        return loop_body

    def run(self, time_step=None, policy_state=None, maximum_iterations=None, high_lvl_action=None):
        """Takes steps in the environment using the policy while updating observers.

        Args:
          time_step: optional initial time_step. If None, it will use the
            current_time_step of the environment. Elements should be shape
            [batch_size, ...].
          policy_state: optional initial state for the policy.
          maximum_iterations: Optional maximum number of iterations of the while
            loop to run. If provided, the cond output is AND-ed with an additional
            condition ensuring the number of iterations executed is no greater than
            maximum_iterations.

        Returns:
          time_step: TimeStep named tuple with final observation, reward, etc.
          policy_state: Tensor with final step policy state.
        """
        return self._run_fn(
            time_step=time_step,
            policy_state=policy_state,
            maximum_iterations=maximum_iterations, high_lvl_action=high_lvl_action)

    # TODO(b/113529538): Add tests for policy_state.
    def _run(self, time_step=None, policy_state=None, maximum_iterations=None, high_lvl_action=None):
        """See `run()` docstring for details."""
        if time_step is None:
            time_step = self.env.current_time_step()
        if policy_state is None:
            policy_state = self.policy.get_initial_state(self.env.batch_size)

        # Batch dim should be first index of tensors during data collection.
        batch_dims = nest_utils.get_outer_shape(time_step,
                                                self.env.time_step_spec())
        counter = tf.zeros(batch_dims, tf.int32)

        [_, time_step, policy_state, _] = tf.nest.map_structure(tf.stop_gradient,
                                                                tf.while_loop(
                                                                    cond=self._loop_condition_fn(),
                                                                    body=self._loop_body_fn(),
                                                                    loop_vars=[counter, time_step, policy_state,
                                                                               high_lvl_action],
                                                                    parallel_iterations=1,
                                                                    maximum_iterations=maximum_iterations,
                                                                    name='driver_loop'))
        return time_step, policy_state


@gin.configurable
class HighLvlDriver(driver.Driver):
    """A driver that takes N steps in an environment using a tf.while_loop.

  The while loop will run num_steps in the environment, only counting steps that
  result in an environment transition, i.e. (time_step, action, next_time_step).
  If a step results in environment resetting, i.e. time_step.is_last() and
  next_time_step.is_first() (traj.is_boundary()), this is not counted toward the
  num_steps.

  As environments run batched time_steps, the counters for all batch elements
  are summed, and execution stops when the total exceeds num_steps. When
  batch_size > 1, there is no guarantee that exactly num_steps are taken -- it
  may be more but never less.

  This termination condition can be overridden in subclasses by implementing the
  self._loop_condition_fn() method.
  """

    def __init__(
            self,
            env,
            policy,
            observers=None,
            transition_observers=None,
            num_steps=1,
    ):
        """Creates a DynamicStepDriver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after every step in the
        environment. Each observer is a callable(time_step.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)).
      num_steps: The number of steps to take in the environment.

    Raises:
      ValueError:
        If env is not a tf_environment.Base or policy is not an instance of
        tf_policy.Base.
    """
        super(HighLvlDriver, self).__init__(env, policy, observers,
                                            transition_observers)
        self._num_steps = num_steps
        self._run_fn = common.function_in_tf1()(self._run)
        self._is_bandit_env = is_bandit_env(env)

    def _loop_condition_fn(self):
        """Returns a function with the condition needed for tf.while_loop."""

        def loop_cond(counter, *_):
            """Determines when to stop the loop, based on step counter.

      Args:
        counter: Step counters per batch index. Shape [batch_size] when
          batch_size > 1, else shape [].

      Returns:
        tf.bool tensor, shape (), indicating whether while loop should continue.
      """
            return tf.less(tf.reduce_sum(input_tensor=counter), self._num_steps)

        return loop_cond

    def _loop_body_fn(self):
        """Returns a function with the driver's loop body ops."""

        def loop_body(counter, time_step, policy_state):
            """Runs a step in environment.

      While loop will call multiple times.

      Args:
        counter: Step counters per batch index. Shape [batch_size].
        time_step: TimeStep tuple with elements shape [batch_size, ...].
        policy_state: Policy state tensor shape [batch_size, policy_state_dim].
          Pass empty tuple for non-recurrent policies.

      Returns:
        loop_vars for next iteration of tf.while_loop.
      """
            action_step = self.policy.action(time_step, policy_state)
            policy_state = action_step.state
            next_time_step = self.env.step(action_step.action)

            if self._is_bandit_env:
                # For Bandits we create episodes of length 1.
                # Since the `next_time_step` is always of type LAST we need to replace
                # the step type of the current `time_step` to FIRST.
                batch_size = tf.shape(input=time_step.discount)
                time_step = time_step._replace(
                    step_type=tf.fill(batch_size, ts.StepType.FIRST))

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            observer_ops = [observer(traj) for observer in self._observers]
            transition_observer_ops = [
                observer((time_step, action_step, next_time_step))
                for observer in self._transition_observers
            ]
            with tf.control_dependencies(
                    [tf.group(observer_ops + transition_observer_ops)]):
                time_step, next_time_step, policy_state = tf.nest.map_structure(
                    tf.identity, (time_step, next_time_step, policy_state))

            # While loop counter should not be incremented for episode reset steps.
            counter += tf.cast(~traj.is_boundary(), dtype=tf.int32)

            return [counter, next_time_step, policy_state]

        return loop_body

    def run(self, time_step=None, policy_state=None, maximum_iterations=None):
        """Takes steps in the environment using the policy while updating observers.

    Args:
      time_step: optional initial time_step. If None, it will use the
        current_time_step of the environment. Elements should be shape
        [batch_size, ...].
      policy_state: optional initial state for the policy.
      maximum_iterations: Optional maximum number of iterations of the while
        loop to run. If provided, the cond output is AND-ed with an additional
        condition ensuring the number of iterations executed is no greater than
        maximum_iterations.

    Returns:
      time_step: TimeStep named tuple with final observation, reward, etc.
      policy_state: Tensor with final step policy state.
    """
        return self._run_fn(
            time_step=time_step,
            policy_state=policy_state,
            maximum_iterations=maximum_iterations)

    # TODO(b/113529538): Add tests for policy_state.
    def _run(self, time_step=None, policy_state=None, maximum_iterations=None):
        """See `run()` docstring for details."""
        if time_step is None:
            time_step = self.env.current_time_step()
        if policy_state is None:
            policy_state = self.policy.get_initial_state(self.env.batch_size)

        # Batch dim should be first index of tensors during data collection.
        batch_dims = nest_utils.get_outer_shape(time_step,
                                                self.env.time_step_spec())
        counter = tf.zeros(batch_dims, tf.int32)

        [_, time_step, policy_state] = tf.while_loop(
            cond=self._loop_condition_fn(),
            body=self._loop_body_fn(),
            loop_vars=[counter, time_step, policy_state],
            back_prop=False,
            parallel_iterations=1,
            maximum_iterations=maximum_iterations,
            name='driver_loop')
        return time_step, policy_state


def create_network(name: str, input_spec, output_spec):
    preprocessing_layers = {
        'image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (3, 3)),
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
