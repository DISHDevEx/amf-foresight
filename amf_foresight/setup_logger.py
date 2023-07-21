from datetime import datetime
import logging
import time
import os

def setup_logger(filename= "log:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join("logs", filename)), logging.StreamHandler()])