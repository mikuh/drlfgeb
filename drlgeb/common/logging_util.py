import logging
import sys

log_stderr = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_stderr)