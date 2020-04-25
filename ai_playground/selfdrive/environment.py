import click
import cv2
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

        self._episode_ended = False
        self._env.reset()

    def observation_spec(self):
        unity_behaviour_spec = self._env.get_behavior_spec(self._env_config['behavior_name'])
        observation_spec = unity_behaviour_spec.observation_shapes[0]
        return ArraySpec(
            observation_spec,
            dtype=np.float32,
            name="observations")

    def action_spec(self):
        unity_behaviour_spec = self._env.get_behavior_spec(self._env_config['behavior_name'])
        action_spec = BoundedArraySpec(
            shape=(1, unity_behaviour_spec.action_shape),
            dtype=np.float32 if unity_behaviour_spec.is_action_continuous() else np.int32,
            minimum=-1,
            maximum=1,
            name="actions"
        )
        return action_spec

    def get_info(self):
        pass

    def _step(self, action):
        if self._episode_ended:
            return self._reset()
        decision_steps, terminal_steps = self._env.get_steps(self._env_config['behavior_name'])
        if len(decision_steps) > 0:
            self._env.set_actions(
                behavior_name=self._env_config['behavior_name'],
                action=action
            )
            self._env.step()
            self._episode_ended = False
            # cv2.imshow("obs", decision_steps[0].obs[0])
            # cv2.waitKey(0)
            return ts.transition(decision_steps[0].obs[0], decision_steps[0].reward)
        else:
            decision_steps, terminal_steps = self._env.get_steps(self._env_config['behavior_name'])
            if len(terminal_steps) > 0:
                self._episode_ended = True
                return ts.termination(terminal_steps[0].obs[0], terminal_steps[0].reward)

        return 1

    def _reset(self):
        self._episode_ended = False
        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self._env_config['behavior_name'])
        return ts.restart(decision_steps[0].obs[0])
