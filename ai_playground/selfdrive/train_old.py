import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import neptune
import tensorflow as tf

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

print(tf.__version__)
# tensorflow HW setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

np.random.seed(1)
tf.random.set_seed(1)

env_name = "unity_envs\\ai"
engine_configuration_channel = EngineConfigurationChannel()
# file_name=None => use the env from editor
env = UnityEnvironment(base_port=5004, file_name=None, side_channels=[engine_configuration_channel])
env.reset()
group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)

engine_configuration_channel.set_configuration_parameters(time_scale=3.0, width=1366, height=768)

step_result = env.get_step_result(group_name)

print("Number of observations : ", len(group_spec.observation_shapes))
print("Agent state looks like: \n{}".format(step_result.obs[0]))
vis_obs = any([len(shape) == 3 for shape in group_spec.observation_shapes])
print("Is there a visual observation ?", vis_obs)
if vis_obs:
    vis_obs_index = next(i for i, v in enumerate(group_spec.observation_shapes) if len(v) == 3)
    print("Agent visual observation look like:")
    obs = step_result.obs[vis_obs_index]
    # plt.imshow(obs[0, :, :, :])
    # cv2.imshow("Hello", obs[0, :, :, :])

for episode in range(10):
    env.reset()
    step_result = env.get_step_result(group_name)
    done = False
    episode_rewards = 0
    while not done:
        action_size = group_spec.action_size
        if group_spec.is_action_continuous():
            action = np.random.randn(step_result.n_agents(), group_spec.action_size)

        if group_spec.is_action_discrete():
            branch_size = group_spec.discrete_action_branches
            action = np.column_stack(
                [np.random.randint(0, branch_size[i], size=(step_result.n_agents())) for i in range(len(branch_size))])
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        # obs = step_result.obs[vis_obs_index]
        # cv2.imshow("Hello", obs[0, :, :, :])
        episode_rewards += step_result.reward[0]
        done = step_result.done[0]
    print("Total reward this episode: {}".format(episode_rewards))
env.close()
