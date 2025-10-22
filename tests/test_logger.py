from src.monitoring.logger import get_logger, log_metric
from src.monitoring.metrics import track_inference_time
import time

logger = get_logger(__name__)


def test_logger_message():
    logger.info("This is a test log message")
    assert True  


def test_metric_logging():
    log_metric("test_accuracy", 0.95, step=1)
    assert True 


@track_inference_time
def dummy_inference():
    time.sleep(0.2)
    return "done"


def test_inference_time_tracking():
    result = dummy_inference()
    assert result == "done"
