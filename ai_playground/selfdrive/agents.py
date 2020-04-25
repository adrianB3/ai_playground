import click


class HAgent:
    def __init__(self, ctx: click.Context):
        self.config = ctx.obj['config']

