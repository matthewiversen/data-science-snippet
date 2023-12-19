# https://gaurav-adarshi.medium.com/logging-module-in-python-a3ed49257928

import logging
import colorlog

# Create a logger
logger = logging.getLogger(__name__)

# Create a ColoredFormatter with the desired format
formatter = colorlog.ColoredFormatter(
    "%(asctime)s %(log_color)s%(levelname)s%(reset)s: %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'white,bg_red',
    },
    secondary_log_colors={},
    style='%'
)

# Create a StreamHandler and set the formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)


logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")