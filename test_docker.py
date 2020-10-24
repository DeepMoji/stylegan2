import sys

import logging
import time


if __name__ == "__main__":

    LOG_FILE = sys.argv[1]

    logging.basicConfig(level=logging.INFO, filename=LOG_FILE)

    for cnt in range(1000):
        time.sleep(1)
        res_string = "Processing image num " + str(cnt) + " "
        logging.info(res_string)
