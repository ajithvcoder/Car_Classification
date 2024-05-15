import os 
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"run_{timestamp}.log"
log_filepath = os.path.join(log_dir, log_filename)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger object
logger = logging.getLogger(__name__)

# Example usage of the logger
logger.info("Logging initialized successfully.")