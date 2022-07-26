import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
LOG_DIR = os.path.join(PROJECT_DIR, "logs")


def generate_logger(caller_package_name, file_name):

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    today = datetime.now().strftime(r"%d%m%Y")
    log_folder = os.path.join(LOG_DIR, caller_package_name, today)
    log_file = os.path.join(log_folder, file_name)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=(10 * 1024 * 1024), backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    log_format = logging.Formatter("%(asctime)s " "%(levelname)s - %(message)s")
    file_handler.setFormatter(log_format)

    logger.addHandler(file_handler)

    return logger
