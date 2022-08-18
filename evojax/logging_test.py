from util import create_logger
import logging

from google.cloud import logging as cloudlogging


if __name__ == "__main__":
    log_dir = "./test_logs"
    logger = create_logger('TEST', log_dir, debug=True)

    logger.info('EvoJAX Logging Test')
    logger.info('Is it working now??')
    logger.debug('Debug test....')

    logging.basicConfig(filename='test.log', encoding='utf-8', level=logging.DEBUG)
    logging.info('EvoJAX Logging Test')
    logging.info('Is it working now??')
    logging.debug('Debug test....')

    lg_client = cloudlogging.Client()

    lg_handler = lg_client.get_default_handler()
    cloud_logger = logging.getLogger("cloudLogger")
    cloud_logger.setLevel(logging.INFO)
    cloud_logger.addHandler(lg_handler)
    cloud_logger.info("test out logger carrying normal news")
    cloud_logger.error("test out logger carrying bad news")


