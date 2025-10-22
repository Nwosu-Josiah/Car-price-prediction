import logging
import sys
from datetime import datetime


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_metric(metric_name: str, value, step: int = None):
    
    log_entry = {
        "metric": metric_name,
        "value": value,
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }
    logging.info(f"METRIC_LOG | {log_entry}")
