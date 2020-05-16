import click
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents import tf_agent
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.trajectories import trajectory

from ai_playground.selfdrive.networks import create_network
from ai_playground.selfdrive.optimizers import get_optimizer


class PPOAgent:
    def __init__(self, ctx: click.Context):
        algo_name = 'ppo'
        self.config = ctx.obj['config']['algorithm']['params'][algo_name]

    def create_ppo_agent(self, tf_env: TFEnvironment, train_step_counter) -> ppo_agent.PPOAgent:
        tf_agents_ppo = ppo_agent.PPOAgent(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=tf_env.action_spec(),
            optimizer=get_optimizer(self.config['optimizer'], self.config['optimizer_params'][self.config['optimizer']]),
            actor_net=create_network(self.config['ppo_actor_net'], tf_env),
            value_net=create_network(self.config['ppo_value_net'], tf_env),
            num_epochs=self.config['num_epochs'],
            train_step_counter=train_step_counter,
            discount_factor=self.config['discount_factor'],
            gradient_clipping=self.config['gradient_clipping'],
            entropy_regularization=self.config['entropy_regularization'],
            importance_ratio_clipping=self.config['importance_ratio_clipping'],
            use_gae=self.config['use_gae'],
            use_td_lambda_return=self.config['use_td_lambda_return']
        )

        return tf_agents_ppo


# class HAgent(tf_agent.TFAgent):
#     """
#     Algorithm description:
#     1.  Pre-train low level skills
#     2.  Initialize high-level policy pi_h with a random policy
#     3.  Initialize k_1 and k_S
#     4.  for i in range 1...N
#             Collect experiences for T low-level steps
#             Modify low-level experience with r_t
#             Optimize pi_h with the high-level experience of the iter i
#             Optimize pi_l with the modified low-level experience for iter i
#             k_i+1 = max(f(k_i), k_s)
#         end for
#     5.  return pi_h, pi_l
#     Needed processes:
#     - Low level skills pre-training (initialization)
#     - Different objective functions from the same experience
#     - Calculation of auxiliary reward from the high level advantage function
#     - Skill length annealing (ks)
#     - Optimization of high and low lvl policies with an optimization algo of choice (on or off policy)
#     in parallel.
#     """
#     def __init__(
#             self, ctx: click.Context,
#             time_step_spec,
#             action_spec,
#             train_sequence_length,
#             name=None
#             ):
#         algo_name = ctx.obj['config']['algorithm']['name']
#         self.config = ctx.obj['config']['algorithm']['params'][algo_name]
#         tf.Module.__init__(name=name)
#         if self.config['policy_optimizer'] == 'ppo':
#             self.h_agent: ppo_agent.PPOAgent = PPOAgent(ctx).create_ppo_agent()
#             self.l_agent: ppo_agent.PPOAgent = PPOAgent(ctx).create_ppo_agent()
#         super().__init__(time_step_spec, action_spec, policy, collect_policy, train_sequence_length)
#
#     def _initialize(self):
#         pass
#
#     def _train(self, experience, weights):
#         # Get individual tensors from transitions.
#         (time_steps, policy_steps_,
#          next_time_steps) = trajectory.to_transition(experience)
#         actions = policy_steps_.action
