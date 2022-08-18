from util import create_logger
import logging


if __name__ == "__main__":
    log_dir = "./test_logs"
    logger = create_logger('TEST', log_dir, debug=True)

    logger.info('EvoJAX Logging Test')
    logger.info('Is it working now??')
    logger.debug('Debug test....')

    logging.info('EvoJAX Logging Test')
    logging.info('Is it working now??')
    logging.debug('Debug test....')


