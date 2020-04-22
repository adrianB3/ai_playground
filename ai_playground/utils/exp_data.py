import neptune
import click
import socket
import datetime
from neptune.git_info import GitInfo
from git import Repo

from ai_playground.utils.logger import get_logger

repo = Repo(path='.', search_parent_directories=True)


def init_neptune(ctx: click.Context):
    project = ctx.obj['config']['utils']['neptune_project']
    neptune.init(project)
    gitInfo = GitInfo(
        commit_id=repo.head.commit,
        message=repo.active_branch.commit.message,
        author_name=repo.head.commit.author.name,
        author_email=repo.head.commit.author.email,
        commit_date=datetime.datetime.utcfromtimestamp(repo.head.commit.committed_date),
        repository_dirty=repo.is_dirty(),
        active_branch=repo.active_branch,
        remote_urls=repo.remotes
    )
    exp = neptune.create_experiment(
        name=ctx.obj['exp_name'],
        params=ctx.obj['config'],
        logger=get_logger(),
        send_hardware_metrics=True,
        upload_source_files=[],
        upload_stdout=False,
        upload_stderr=False,
        # git_info=gitInfo,
        hostname=socket.gethostname()
    )
    return exp
