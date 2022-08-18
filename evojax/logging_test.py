import logging
from util import create_logger

# Imports the Cloud Logging client library
import google.cloud.logging

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
client.setup_logging(log_level=logging.DEBUG)


if __name__ == "__main__":
    log_dir = "./test_logs"
    logger = create_logger('TEST', log_dir, debug=True)

    logger.info('EvoJAX Logging Test')
    logger.info('Is it working now??')
    logger.debug('Debug test....')
    logger.warning('Has the debug log appeared??')


