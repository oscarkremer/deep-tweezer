import os
import sys
import logging

logging.getLogger('numpy').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('keras').setLevel(logging.CRITICAL)
logging.getLogger('sklearn').setLevel(logging.CRITICAL)


class Logger(object):
    def __init__(self, log_file='data/logs/output.log'):   
        self.log_file = os.path.join(os.path.dirname(__file__), '../../data/logs/output.log')
        self.console = logging.StreamHandler(sys.stdout)
        self.console.setLevel(logging.INFO)
        self.formatter = logging.Formatter('---> %(message)s')
        self.console.setFormatter(self.formatter)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=self.log_file,
            filemode='a')

    def log(self, lvl, msg):
        if 'LOGGER' in os.environ and os.environ['LOGGER'] == 'stdout':
            logging.getLogger('').addHandler(self.console)
        logging.log(lvl, msg)

logger = Logger()

def log(msg, lvl=logging.INFO):
    logger.log(lvl, msg)

def label_log(labels):
    sentence = ''
    for label in labels:
        sentence += '-' + label
    return sentence
