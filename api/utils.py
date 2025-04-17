import os
import pickle
import logging
import time
import string
import random
import SimpleITK as sitk
from pathlib import Path


def create_directories_and_files():
    """
    Create all directories and files needed for the application to run that do not already exist.
    :return:
    """
    logger = logging.getLogger('api')
    workdir = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(f'{workdir}/data/'):
        Path(f'{workdir}/data/').mkdir(parents=True, exist_ok=False)
        logger.info('Created data directory.')

    if not os.path.exists(f'{workdir}/uploads/'):
        Path(f'{workdir}/uploads/').mkdir(parents=True, exist_ok=False)
        logger.info('Created uploads directory.')


def init_logger(name: str, level: int = logging.DEBUG):
    """
    Initialise a logger for the application.
    :param name: The context name of the logger.
    :param level: The logging level of the logger.
    :return:
    """
    workdir = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(f'{workdir}/logs/'):
        Path(f'{workdir}/logs/').mkdir(parents=True, exist_ok=False)  # create logs directory if it does not exist

    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.StreamHandler()
    # fh = logging.FileHandler(filename=f'{workdir}/logs/app_log_{time.asctime()}.log', mode='w')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(filename)s - %(funcName)s - %(message)s"))
    logger.addHandler(fh)
    logger.info('Logger initialised.')


