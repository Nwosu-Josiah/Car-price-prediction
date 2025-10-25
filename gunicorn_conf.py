import multiprocessing
import os



cores = multiprocessing.cpu_count()
if cores <= 1:
    workers = 1
elif cores == 2:
    workers = 3
elif cores <= 4:
    workers = 4
else:
    workers = min(cores, 6)
threads = 2
timeout = 120  # seconds
graceful_timeout = 30
loglevel = "info"
accesslog = "-"  
errorlog = "-"
preload_app = True  
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
