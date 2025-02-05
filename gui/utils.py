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
    logger = logging.getLogger('app')
    if not os.path.exists('./session_ids.pickle'):
        with open('./session_ids.pickle', 'wb') as f:
            pickle.dump(set(), f)  # if no session id collection exists, create an empty one
            logger.info('No session id collection found. Created a new one.')

    if not os.path.exists('./data/'):
        Path('./data/').mkdir(parents=True, exist_ok=False)
        logger.info('Created data directory.')

    if not os.path.exists('./uploads/'):
        Path('./uploads/').mkdir(parents=True, exist_ok=False)
        logger.info('Created uploads directory.')


def generate_session_id():
    """
    Generate a unique session id for the user.
    :return: A unique session id.
    """
    logger = logging.getLogger('app')
    with open('./session_ids.pickle', 'rb') as f:
        session_ids = pickle.load(f)

    length = 6

    letters = string.ascii_lowercase
    session_id = ''.join(random.choice(letters) for i in range(length))
    while session_id in session_ids:
        session_id = ''.join(random.choice(letters) for i in range(length))

    session_ids.add(session_id)

    with open('./session_ids.pickle', 'wb') as f:
        pickle.dump(session_ids, f)
        logger.info(f'Session id {session_id} generated.')

    return session_id


def init_logger(name: str, level: int = logging.DEBUG):
    """
    Initialise a logger for the application.
    :param name: The context name of the logger.
    :param level: The logging level of the logger.
    :return:
    """
    if not os.path.exists('./logs/'):
        Path('./logs/').mkdir(parents=True, exist_ok=False)  # create logs directory if it does not exist

    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(filename=f'./logs/app_log_{time.asctime()}.log', mode='w')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(filename)s - %(funcName)s - %(message)s"))
    logger.addHandler(fh)
    logger.info('Logger initialised.')


