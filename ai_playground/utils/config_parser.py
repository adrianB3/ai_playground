import yaml
import os

from ai_playground.utils.logger import get_logger

logger = get_logger()


def load_cfg(yaml_filepath):
    with open(yaml_filepath, 'r') as stream:
        loader = yaml.Loader
        cfg = yaml.load(stream, Loader=loader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)

    return cfg


def make_paths_absolute(dir_, cfg):
    for key in cfg.keys():
        if key.endswith('_path') and cfg[key] is not None:
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logger.error("%s does not exist.", cfg[key])
                raise OSError("Path does not exist.")
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
