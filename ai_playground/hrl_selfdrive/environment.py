import click
import os
import numpy as np
from tf_agents.environments import py_environment
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from tf_agents.specs import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts
from ai_playground.utils.logger import get_logger

logger = get_logger()


def init_unity_env(ctx: click.Context):
    env_cfg = ctx.obj['config']['environment']
    env_path = env_cfg['env_path']
    if env_path is None:
        logger.warning("env_path is None, running in editor")
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(
        time_scale=env_cfg['time_scale'],
        width=1366,
        height=768
    )
    env = UnityEnvironment(
        base_port=5004,
        file_name=env_path,
        side_channels=[engine_configuration_channel],
        no_graphics=False
    )
    return env


class SelfDriveEnvironment(py_environment.PyEnvironment):
    def __init__(self, ctx: click.Context):
        super().__init__()
        self._env_config = ctx.obj['config']['environment']
        self._env: UnityEnvironment = init_unity_env(ctx)
        self._unity_behaviour_spec = self._env.get_behavior_spec(self._env_config['behavior_name'])
        self._episode_ended = False
        self._env.reset()

    def observation_spec(self):
        observation_spec = self._unity_behaviour_spec.observation_shapes
        return ArraySpec(observation_spec, dtype=np.float32, name="observations")

    def action_spec(self):
        action_spec = BoundedArraySpec(
            shape=(self._unity_behaviour_spec.action_shape,),
            dtype=np.float32 if self._unity_behaviour_spec.is_action_continuous() else np.int32,
            minimum=0,
            maximum=self._unity_behaviour_spec.action_size,
            name="actions"
        )
        return action_spec

    def get_info(self):
        pass

    def _step(self, action):
        if self._episode_ended:
            return self._reset()
        self._env.set_actions(
            behavior_name=self._env_config['behavior_name'],
            action=action
        )
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self._env_config['behavior_name'])



    def _reset(self):
        self._episode_ended = False
        self._env.reset()
