import neptune
import click
import logging
from ai_playground.utils.logger import get_logger


def init_neptune(ctx: click.Context):
    project = ctx.obj['config']['neptune_project']
    neptune.init(project)
    exp = neptune.create_experiment(
        name=ctx.obj['exp_name'],
        params=ctx.obj['config'],
        logger=get_logger(),
        send_hardware_metrics=True,
        upload_source_files=[],
        upload_stdout=False,
        upload_stderr=False
    )
    return exp
