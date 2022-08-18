from util import create_logger


if __name__ == "__main__":
    log_dir = "./test_logs"
    logger = create_logger('TEST', log_dir, debug=True)

    logger.info('EvoJAX Logging Test')
    logger.info('Is it working now??')
    logger.debug('Debug test....')


