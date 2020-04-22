import click
import pprint
import neptune
from pyfiglet import Figlet
from datetime import datetime
from ai_playground.utils.logger import get_logger
from ai_playground.utils.config_parser import load_cfg
from ai_playground.utils.exp_data import init_neptune
from ai_playground.hrl_selfdrive.environment import SelfDriveEnvironment
logger = get_logger()


@click.group()
@click.option('--exp_name', default='exp_' + datetime.now().strftime("%d/%m/%Y_%H:%M:%S"), help='The experiment name.')
@click.option('--config', help='Training configuration file.')
@click.option('--log2neptune/--no-log2neptune', default=False, help='Choose whether to log to neptune or not.')
@click.pass_context
def main(ctx, config: str, log2neptune: bool, exp_name: str):
    figlet = Figlet(font='slant')
    click.echo(figlet.renderText("AI Playground"))
    cfg = load_cfg(config)
    ctx.obj['exp_name'] = exp_name
    ctx.obj['config_file'] = config
    ctx.obj['config'] = cfg
    ctx.obj['log2neptune'] = log2neptune
    if log2neptune:
        exp = init_neptune(ctx)
        logger.info("This experiment is logged to neptune.ai")


@main.command()
@click.pass_context
def train(ctx):
    logger.info("Loaded config: " + ctx.obj['config_file'])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict(ctx.obj['config']))

    logger.info("Starting training")
    sfd = SelfDriveEnvironment(ctx)
    act = sfd.action_spec()
    print()
    if ctx.obj['log2neptune']:
        neptune.stop()


def start():
    main(obj={})


if __name__ == '__main__':
    start()
