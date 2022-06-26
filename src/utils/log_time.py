import time
from .logger import log
from datetime import timedelta
from contextlib import contextmanager

@contextmanager
def log_time(title):
    start = time.time()
    log('Started ' + title)
    try:
        yield
    finally:
        log('Done {0} in {1}'.format(title, str(timedelta(seconds=time.time() - start))))