import logging

logger = logging.getLogger("hedata")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
