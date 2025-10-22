

import time
from functools import wraps
from src.monitoring.logger import log_metric


def track_inference_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = round(time.time() - start, 4)
        log_metric("inference_latency_seconds", latency)
        return result

    return wrapper


def track_training_duration(func):
   
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = round(time.time() - start, 2)
        log_metric("training_duration_seconds", duration)
        return result

    return wrapper
