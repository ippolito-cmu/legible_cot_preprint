
import logging
import os
from os.path import *
from datetime import datetime
import getpass
import socket
import platform
import sys
import torch
def get_logger(output_dir: str, args) -> logging.Logger:
    """
    Set up logging with comprehensive metadata about the run.
    Creates both console and file handlers.
    Logs are saved in output_dir/logs/
    """
    log_dir = output_dir + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir + f"/run_{timestamp}.log"
    logger = logging.getLogger("vllm_self_preference")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("=" * 80)
    logger.info("NEW RUN STARTED")
    logger.info("=" * 80)
    logger.info("SYSTEM METADATA:")
    logger.info(f"  User: {getpass.getuser()}")
    logger.info(f"  Hostname: {socket.gethostname()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {sys.version.split()[0]}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
    logger.info("RUN PARAMETERS:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info("=" * 80)
    return logger
