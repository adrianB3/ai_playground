import click
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.environments.tf_environment import TFEnvironment

from ai_playground.selfdrive.networks import create_network
from ai_playground.selfdrive.optimizers import get_optimizer


class PPOAgent:
    def __init__(self, ctx: click.Context):
        algo_name = ctx.obj['config']['algorithm']['name']
        self.config = ctx.obj['config']['algorithm']['params'][algo_name]

    def create_ppo_agent(self, tf_env: TFEnvironment, train_step_counter) -> TFAgent:
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


class HAgent(TFAgent):
    def __init__(self, ctx: click.Context, time_step_spec, action_spec, policy, collect_policy, train_sequence_length):
        super().__init__(time_step_spec, action_spec, policy, collect_policy, train_sequence_length)
        self.config = ctx.obj['config']

    def _initialize(self):
        pass

    def _train(self, experience, weights):
        pass
