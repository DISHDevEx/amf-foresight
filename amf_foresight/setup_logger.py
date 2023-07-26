from datetime import datetime
import logging
import time
import os

def setup_logger(filename= "log:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')):
    """
    This function sets up the logger that will be used to log all events in the system when the script is running.

    Args:
        filename (str, optional): The name of the log file. Defaults to a 
        timestamped log file ("log:YYYY-mm-dd HH:MM:SS").

    Creates a 'logs' directory, if it doesn't exist, to store all log files.
    Configures the logging module to log all messages of level INFO and above.
    Messages are logged to the console and the file.
    
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join("logs", filename)), logging.StreamHandler()])