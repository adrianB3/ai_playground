import logging
import colorlog


def get_logger(module) -> logging.Logger:
    log_format = (
        '[%(asctime)s] '
        '[%(levelname)s] '
        '%(name)s.'
        '%(funcName)s : '
        '%(message)s'
    )
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)

    # Output full log
    fh = logging.FileHandler('ai_playground.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # # Output warning log
    # fh = logging.FileHandler('ai_playground.warning.log')
    # fh.setLevel(logging.WARNING)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    #
    # # Output error log
    # fh = logging.FileHandler('ai_playground.error.log')
    # fh.setLevel(logging.ERROR)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    return logger
